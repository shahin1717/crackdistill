"""
KD Trainer — correct implementation
=====================================
Processes soft logits and intermediate encoder features from SAM.
Registers hooks and trains 1x1 projection convolutions for feature distillation.
Uses picklable hooks and temporary hook/loss restoration during model saving to prevent pickling errors.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict
from ultralytics.models.yolo.segment.train import SegmentationTrainer


class KDYOLODataset(torch.utils.data.Dataset):
    """
    Wrapper for YOLO Dataset that preloads SAM teacher logits and features
    inside dataloader worker processes to hide disk I/O latency from GPU training.
    """
    def __init__(self, base_dataset, logits_dir, kd_cfg):
        self.base_dataset = base_dataset
        self.logits_dir = Path(logits_dir)
        self.features_dir = self.logits_dir.parent / "teacher_features"
        self.kd_cfg = kd_cfg

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img_path = item.get("im_file", "")
        if not img_path:
            item["sam_target"] = None
            item["sam_feat"] = None
            return item
            
        stem = Path(img_path).stem

        sam_target = None
        sam_feat = None

        for prefix in [f"crack500_{stem}", f"deepcrack_{stem}", stem]:
            c = self.logits_dir / f"{prefix}_logits.npy"
            f_path = self.features_dir / f"{prefix}_features.npz"

            if c.exists():
                try:
                    raw = np.load(str(c))  # shape (M, 256, 256)
                    if raw.ndim == 2:
                        raw = np.expand_dims(raw, axis=0)
                    sam_target = torch.from_numpy(raw).float()
                except Exception:
                    pass

                if self.kd_cfg.losses.feature.enabled:
                    if not f_path.exists():
                        f_path = self.logits_dir / f"{prefix}_features.npz"
                    if f_path.exists():
                        try:
                            with np.load(str(f_path)) as data:
                                sam_feat = {
                                    "image_embed": torch.from_numpy(data["image_embed"]).float(),
                                    "feat1": torch.from_numpy(data["feat1"]).float()
                                }
                        except Exception:
                            pass
                break

        item["sam_target"] = sam_target
        item["sam_feat"] = sam_feat
        return item

    @property
    def collate_fn(self):
        return self.base_dataset.collate_fn

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)


class ActiveHook:
    """
    A top-level picklable hook callback class.
    Writes features directly to a class attribute to avoid referencing local closures.
    """
    def __init__(self, key):
        self.key = key

    def __call__(self, module, input, output):
        KDSegmentationTrainer.student_features[self.key] = output


class KDSegmentationTrainer(SegmentationTrainer):
    # Class-level attribute to store hooked features safely
    student_features = {}

    def __init__(self, cfg=None, overrides=None, _callbacks=None, logits_dir=None, kd_cfg=None, **kwargs):
        import os
        from pathlib import Path

        # Auto-convert master ConfigNode / master dict into Ultralytics cfg if master config passed directly
        if hasattr(cfg, "student") or (isinstance(cfg, dict) and "student" in cfg):
            master_cfg = cfg
            if kd_cfg is None:
                if hasattr(master_cfg, "distillation"):
                    kd_cfg = master_cfg.distillation
                elif isinstance(master_cfg, dict) and "distillation" in master_cfg:
                    kd_cfg = master_cfg["distillation"]

            if logits_dir is None:
                if hasattr(master_cfg, "teacher") and hasattr(master_cfg.teacher, "logits_dir"):
                    logits_dir = master_cfg.teacher.logits_dir
                elif isinstance(master_cfg, dict) and "teacher" in master_cfg and "logits_dir" in master_cfg["teacher"]:
                    logits_dir = master_cfg["teacher"]["logits_dir"]

            # Backbone model
            model_name = "yolo11n-seg"
            if hasattr(master_cfg, "student") and hasattr(master_cfg.student, "backbone"):
                model_name = master_cfg.student.backbone
            elif isinstance(master_cfg, dict) and "student" in master_cfg and "backbone" in master_cfg["student"]:
                model_name = master_cfg["student"]["backbone"]
            if not str(model_name).endswith(".pt"):
                model_name = f"{model_name}.pt"

            # Data YAML path
            data_path = "data/datasets/crack500_yolo/dataset.yaml"
            if hasattr(master_cfg, "data"):
                if hasattr(master_cfg.data, "datasets") and len(master_cfg.data.datasets) > 0:
                    d_p = master_cfg.data.datasets[0].path
                    candidate = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")
                    if os.path.exists(candidate):
                        data_path = candidate
                    elif os.path.exists("data/datasets/crack500_yolo/dataset.yaml"):
                        data_path = "data/datasets/crack500_yolo/dataset.yaml"
                elif hasattr(master_cfg.data, "path"):
                    d_p = master_cfg.data.path
                    data_path = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")
            elif isinstance(master_cfg, dict) and "data" in master_cfg:
                d_dict = master_cfg["data"]
                if "datasets" in d_dict and len(d_dict["datasets"]) > 0:
                    d_p = d_dict["datasets"][0].get("path", data_path) if isinstance(d_dict["datasets"][0], dict) else getattr(d_dict["datasets"][0], "path", data_path)
                    data_path = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")

            check_data = (overrides.get("data") if overrides and isinstance(overrides, dict) else None) or data_path
            if check_data and not Path(check_data).exists():
                alt = Path("data/datasets/combined_yolo/dataset.yaml")
                if alt.exists():
                    check_data = str(alt)
                    data_path = check_data
                    if overrides and isinstance(overrides, dict) and "data" in overrides:
                        overrides["data"] = check_data
                elif os.path.exists("data/datasets/crack500_yolo/dataset.yaml"):
                    check_data = "data/datasets/crack500_yolo/dataset.yaml"
                    data_path = check_data
                    if overrides and isinstance(overrides, dict) and "data" in overrides:
                        overrides["data"] = check_data

            # Safely fix hardcoded absolute paths in dataset.yaml.
            # The dataset folder may be a symlink to a read-only /kaggle/input path,
            # so we NEVER write to the symlink target.
            # Instead, we copy dataset.yaml to a writable local path and use that copy.
            if check_data and Path(check_data).exists():
                try:
                    import shutil as _shutil
                    yaml_p = Path(check_data)
                    # Resolve the ACTUAL target to check if it is read-only
                    real_yaml = yaml_p.resolve()
                    abs_dset_dir = str(yaml_p.parent.resolve())
                    original_text = real_yaml.read_text()
                    fixed_lines = [f"path: {abs_dset_dir}" if l.strip().startswith("path:") else l for l in original_text.splitlines()]
                    fixed_text = "\n".join(fixed_lines) + "\n"
                    # Only write if the text actually changed
                    if fixed_text != original_text:
                        # Try writing in place (works if writable)
                        try:
                            yaml_p.write_text(fixed_text)
                            print(f"[KD] Fixed dataset.yaml path in-place: {abs_dset_dir}")
                        except OSError:
                            # Read-only filesystem (Kaggle input symlink): copy to local writable dir
                            local_yaml_dir = Path("data/datasets_yaml")
                            local_yaml_dir.mkdir(parents=True, exist_ok=True)
                            local_yaml = local_yaml_dir / yaml_p.parent.name / "dataset.yaml"
                            local_yaml.parent.mkdir(parents=True, exist_ok=True)
                            local_yaml.write_text(fixed_text)
                            data_path = str(local_yaml)
                            if overrides and isinstance(overrides, dict) and "data" in overrides:
                                overrides["data"] = str(local_yaml)
                            print(f"[KD] Read-only symlink — copied fixed dataset.yaml to: {local_yaml}")
                except Exception as e:
                    print(f"[KD Warning] dataset.yaml path fix skipped: {e}")

            proj_name = getattr(getattr(master_cfg, "project", None), "name", "runs")
            exp_name = getattr(getattr(master_cfg, "project", None), "experiment", "exp")

            auto_overrides = {
                "model": model_name,
                "data": data_path,
                "epochs": getattr(getattr(master_cfg, "train", None), "epochs", 150),
                "imgsz": getattr(getattr(master_cfg, "student", None), "imgsz", getattr(getattr(master_cfg, "data", None), "image_size", 512)),
                "batch": getattr(getattr(master_cfg, "data", None), "batch_size", 16),
                "amp": getattr(getattr(master_cfg, "train", None), "amp", False),
                "lr0": getattr(getattr(master_cfg, "train", None), "lr", 0.001),
                "weight_decay": getattr(getattr(master_cfg, "train", None), "weight_decay", 0.0005),
                "project": str(proj_name),
                "name": str(exp_name),
                "exist_ok": True,
                "task": "segment",
            }
            if overrides and isinstance(overrides, dict):
                auto_overrides.update(overrides)

            from ultralytics.cfg import get_cfg
            cfg = get_cfg(overrides=auto_overrides)
            overrides = None

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks, **kwargs)

        if logits_dir is None:
            logits_dir = os.environ.get("KD_LOGITS_DIR", "data/teacher_logits/")
                
        if kd_cfg is None:
            kd_config_json = os.environ.get("KD_CONFIG")
            if kd_config_json:
                try:
                    import json
                    from utils.config_loader import ConfigNode
                    kd_cfg = ConfigNode(json.loads(kd_config_json))
                except Exception:
                    pass
            
            if kd_cfg is None:
                # Under DDP, we can load configuration dynamically as a fallback
                from utils.config_loader import load_config
                try:
                    full_cfg = load_config("configs/config.yaml")
                    kd_cfg = full_cfg.distillation
                except Exception:
                    pass

        self.logits_dir  = Path(str(logits_dir))
        self.kd_cfg      = kd_cfg
        
        if kd_cfg is not None:
            self.temperature = float(kd_cfg.temperature)
            self.kd_weight   = float(kd_cfg.losses.boundary.weight)
        else:
            self.temperature = 1.6502
            self.kd_weight   = 2.0569
        
        self._current_paths = []
        self._kd_logged  = False
        self._no_logits_warned = False
        self.kd_losses   = []
        self._sam_targets = {}   # image_stem → soft target tensor (M, 256, 256)
        self._sam_features = {}  # image_stem → dict of features
        self._hook_handles = []

        logit_files = list(self.logits_dir.glob("*.npy"))
        if len(logit_files) == 0:
            for candidate in [Path("/tmp") / self.logits_dir.name, Path("/tmp/teacher_logits_box"), Path("/tmp/teacher_logits")]:
                if candidate.exists():
                    c_files = list(candidate.glob("*.npy"))
                    if len(c_files) > 0:
                        print(f"[KD] Found {len(c_files)} logit files in fallback '{candidate}'. Redirecting logits_dir.")
                        self.logits_dir = candidate
                        logit_files = c_files
                        try:
                            p_local = Path(str(logits_dir))
                            if not p_local.exists() or (p_local.is_dir() and not list(p_local.glob("*.npy"))):
                                if os.path.lexists(p_local):
                                    os.unlink(p_local) if os.path.islink(p_local) else shutil.rmtree(p_local)
                                p_local.parent.mkdir(parents=True, exist_ok=True)
                                os.symlink(candidate, p_local)
                                print(f"[KD] Created symlink: {p_local} -> {candidate}")
                        except Exception as sym_err:
                            print(f"[KD Warning] Could not symlink fallback: {sym_err}")
                        break

        print(f"[KD] logits_dir : {self.logits_dir}")
        print(f"[KD] logit files: {len(logit_files)}")
        print(f"[KD] temperature: {self.temperature}")

        is_kd_enabled = hasattr(self.kd_cfg, "enabled") and getattr(self.kd_cfg, "enabled")
        if is_kd_enabled and len(logit_files) == 0:
            raise RuntimeError(
                f"[KD FATAL ERROR] logits_dir '{self.logits_dir}' contains 0 logit files (*.npy)!\n"
                f"Knowledge distillation cannot proceed without precomputed teacher logits.\n"
                f"Please ensure the teacher logits dataset is attached and linked properly to '{self.logits_dir}'."
            )

    def setup_model(self):
        """Build model, set up projection layers and hooks, and call parent setup."""
        head_idx = 22
        is_freeze_head = hasattr(self.kd_cfg, "progressive") and self.kd_cfg.progressive.get("freeze_head", False)

        if is_freeze_head:
            if self.args.freeze is None:
                self.args.freeze = [head_idx]
            elif isinstance(self.args.freeze, list):
                if head_idx not in self.args.freeze:
                    self.args.freeze.append(head_idx)
            elif isinstance(self.args.freeze, int):
                self.args.freeze = list(range(self.args.freeze))
                if head_idx not in self.args.freeze:
                    self.args.freeze.append(head_idx)
            print(f"[KD] Progressive: Freezing Segment head at index {head_idx}. args.freeze={self.args.freeze}")

        ckpt = super().setup_model()

        if is_freeze_head:
            try:
                from ultralytics.utils.torch_utils import unwrap_model
                model = unwrap_model(self.model)
            except Exception:
                model = self.model

            for idx, module in enumerate(model.model):
                if type(module).__name__ == "Segment":
                    head_idx = idx
                    break

            for name, param in model.named_parameters():
                if f"model.{head_idx}." in name:
                    param.requires_grad = False
            print(f"[KD] Explicitly set requires_grad=False for Segment head parameters (layer {head_idx})")

        self._setup_proj_layers_and_hooks()
        self._patch_model_loss()
        return ckpt

    def _setup_proj_layers_and_hooks(self):
        """
        Dynamically determine student backbone feature shapes, initialize 1x1 convs
        for channel alignment, and register active training hooks.
        """
        try:
            from ultralytics.utils.torch_utils import unwrap_model
            model = unwrap_model(self.model)
        except Exception:
            model = self.model

        # Determine device
        device = next(self.model.parameters()).device

        # Standard layers to monitor
        layers_to_monitor = self.kd_cfg.losses.feature.layers if hasattr(self.kd_cfg.losses.feature, "layers") else [2, 5, 8]

        captured_shapes = {}
        def temp_hook(layer_idx):
            def hook(module, input, output):
                captured_shapes[layer_idx] = output.shape
            return hook

        hooks = []
        for idx in layers_to_monitor:
            if idx < len(model.model):
                h = model.model[idx].register_forward_hook(temp_hook(idx))
                hooks.append(h)

        # Run dummy forward pass to extract shapes
        dummy_input = torch.zeros((1, 3, self.args.imgsz, self.args.imgsz), device=device)
        model.eval()
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"[KD] Error during dummy forward pass for shapes: {e}")
        model.train()

        # Remove temporary hooks
        for h in hooks:
            h.remove()

        # Build projection layers
        proj_dict = nn.ModuleDict()
        for idx in layers_to_monitor:
            if idx in captured_shapes:
                in_channels = captured_shapes[idx][1]
                feature_h = captured_shapes[idx][2]
                stride = self.args.imgsz // feature_h
                out_channels = 64 if stride <= 4 else 256
                
                # Create a 1x1 Conv to align channels
                proj_dict[f"layer_{idx}"] = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                print(f"[KD] Feature projection layer {idx}: stride {stride}, channels {in_channels} -> {out_channels}")

        # Register projection layers on the model so they are part of optimizer parameters
        model.add_module("proj_layers", proj_dict)
        self.proj_layers = proj_dict.to(device)

        # Register active training hooks
        self._register_active_hooks(model)

    def _register_active_hooks(self, model):
        """Helper to register forward hooks on target student model layers."""
        self._hook_handles.clear()
        KDSegmentationTrainer.student_features.clear()
        
        layers_to_monitor = self.kd_cfg.losses.feature.layers if hasattr(self.kd_cfg.losses.feature, "layers") else [2, 5, 8]
        for idx in layers_to_monitor:
            if idx < len(model.model):
                h = model.model[idx].register_forward_hook(ActiveHook(f"layer_{idx}"))
                self._hook_handles.append(h)
                print(f"[KD] Forward hook registered for layer {idx}")

    def save_model(self):
        """Override save_model to temporarily detach hooks and restore original loss function on both model and EMA model."""
        try:
            from ultralytics.utils.torch_utils import unwrap_model
            model = unwrap_model(self.model)
        except Exception:
            model = self.model

        # Get EMA model if defined
        ema_model = None
        if hasattr(self, "ema") and self.ema is not None and hasattr(self.ema, "ema"):
            try:
                ema_model = unwrap_model(self.ema.ema)
            except Exception:
                ema_model = self.ema.ema

        # Remove hooks on self.model
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        # Recursively clear forward hooks in all submodules for both models
        for m in [model, ema_model]:
            if m is not None:
                for submodule in m.modules():
                    submodule._forward_hooks.clear()

        # Restore original loss function if patched on both models
        original_loss_restored = False
        for m in [model, ema_model]:
            if m is not None and hasattr(m, "original_loss"):
                m.loss = m.original_loss
                original_loss_restored = True

        # Call original saving logic
        result = super().save_model()

        # Re-patch loss function on training model
        if original_loss_restored:
            self._patch_model_loss()

        # Re-register active hooks on training model
        self._register_active_hooks(model)
        
        # Reset the active hooks registered flag so that they get registered on the active training model again
        self._active_hooks_registered = False
        return result

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build custom KD dataset that wraps the default YOLO dataset."""
        base_dataset = super().build_dataset(img_path, mode, batch)
        if mode != "train":
            return base_dataset
        return KDYOLODataset(base_dataset, self.logits_dir, self.kd_cfg)

    def preprocess_batch(self, batch):
        """Preprocess batch and map preloaded SAM targets/features to GPU."""
        # Ensure active hooks are registered on the active running model (handles DDP deepcopy recreation)
        if not hasattr(self, "_active_hooks_registered") or not self._active_hooks_registered:
            try:
                from ultralytics.utils.torch_utils import unwrap_model
                active_model = unwrap_model(self.model)
            except Exception:
                active_model = self.model
            self._register_active_hooks(active_model)
            self._active_hooks_registered = True

        # Clear student features at the start of batch preprocessing
        KDSegmentationTrainer.student_features.clear()

        # Safety check to verify that all projection layer parameters are in the optimizer
        if not hasattr(self, "_checked_optimizer") and hasattr(self, "optimizer") and self.optimizer is not None:
            self._checked_optimizer = True
            proj_params = set(self.proj_layers.parameters())
            opt_params = set()
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    opt_params.add(p)
            missing = proj_params - opt_params
            if missing:
                print(f"[KD] WARNING: {len(missing)} projection layer parameters are NOT in the optimizer! Training them will have no effect.")
            else:
                print("[KD] Success: All projection layer parameters are in the optimizer and will receive gradients.")

        batch = super().preprocess_batch(batch)
        im_files = batch.get("im_file", [])
        if isinstance(im_files, (str, Path)):
            im_files = [im_files]
        self._current_paths = list(im_files)

        # Retrieve the preloaded SAM targets and features from the batch dict
        sam_targets_list = batch.get("sam_target", [])
        sam_feats_list = batch.get("sam_feat", [])

        self._sam_targets = {}
        self._sam_features = {}

        for idx, img_path in enumerate(self._current_paths):
            stem = Path(img_path).stem
            
            if idx < len(sam_targets_list) and sam_targets_list[idx] is not None:
                # Clean NaNs and Infs to prevent nan mask_kd losses
                self._sam_targets[stem] = torch.nan_to_num(sam_targets_list[idx].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                
            if idx < len(sam_feats_list) and sam_feats_list[idx] is not None:
                self._sam_features[stem] = {
                    k: torch.nan_to_num(v.to(self.device), nan=0.0, posinf=0.0, neginf=0.0) for k, v in sam_feats_list[idx].items()
                }

        if self._current_paths and not self._sam_targets and not self._no_logits_warned:
            stems = [Path(p).stem for p in self._current_paths[:3]]
            print(f"[KD] Warning: no SAM logits matched batch stems {stems}. "
                  f"Run: python scripts/generate_teacher_logits.py")
            self._no_logits_warned = True

        return batch

    def _patch_model_loss(self):
        """Patch model.loss() to add KD loss using student predictions."""
        trainer_ref = self

        try:
            from ultralytics.utils.torch_utils import unwrap_model
            model = unwrap_model(self.model)
        except Exception:
            model = self.model

        if not hasattr(model, "original_loss"):
            model.original_loss = model.loss

        original_loss_fn = model.original_loss.__func__ if hasattr(model.original_loss, "__func__") else None

        def patched_loss(self_model, batch, preds=None):
            if preds is None:
                preds = self_model.forward(batch["img"])

            if original_loss_fn is not None:
                base_loss, loss_items = original_loss_fn(self_model, batch, preds)
            else:
                base_loss, loss_items = type(self_model).loss(self_model, batch, preds)

            if not self_model.training:
                return base_loss, loss_items

            kd_losses = trainer_ref._kd_loss_from_preds(preds, batch, self_model)
            
            # Combine losses
            total = base_loss
            for k, v in kd_losses.items():
                total = total + v

            return total, loss_items

        import types
        model.loss = types.MethodType(patched_loss, model)
        print("[KD] model.loss() patched with detailed KD losses ✓")

    def _kd_loss_from_preds(self, preds, batch, model) -> dict:
        """
        Compute KL divergence, boundary, and feature alignment losses.
        """
        kd_losses = {}
        if not self._sam_targets:
            return kd_losses

        try:
            criterion = model.criterion
            preds_parsed = criterion.parse_output(preds)
            
            # Retrieve target assignments
            (fg_mask, target_gt_idx, target_bboxes, _, _), _, _ = criterion.get_assigned_targets_and_loss(preds_parsed, batch)
            
            pred_masks = preds_parsed["mask_coefficient"].permute(0, 2, 1).contiguous()
            proto = preds_parsed["proto"]
            
            loss_mask_kd = torch.tensor(0.0, device=self.device)
            loss_mask_kd_count = 0
            
            loss_boundary = torch.tensor(0.0, device=self.device)
            loss_boundary_count = 0

            # 1. Compute L_mask and L_boundary (Per-instance matched)
            for i, img_path in enumerate(self._current_paths):
                stem = Path(img_path).stem
                if stem not in self._sam_targets:
                    continue
                
                sam_logits = self._sam_targets[stem]
                fg_mask_i = fg_mask[i]
                
                if fg_mask_i.any() and sam_logits.shape[0] > 0:
                    mask_idx = target_gt_idx[i][fg_mask_i]
                    mask_idx = torch.clamp(mask_idx, 0, sam_logits.shape[0] - 1)
                    
                    # Compute student instance predicted mask logits: (N_pos, H_proto, W_proto)
                    pred_coefs = pred_masks[i][fg_mask_i]
                    pred_mask_logits = torch.einsum("in,nhw->ihw", pred_coefs, proto[i])
                    
                    # Extract corresponding SAM teacher logits: (N_pos, 256, 256)
                    sam_logits_matched = sam_logits[mask_idx]
                    
                    # Target resolution (default 256x256, or 512x512 if high_res enabled)
                    target_res = 512 if getattr(self.kd_cfg.losses.mask_kd, "high_res", False) else 256
                    
                    # Resize both to target resolution
                    student_mask_logits_resized = F.interpolate(
                        pred_mask_logits.unsqueeze(1),
                        size=(target_res, target_res),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    
                    sam_logits_matched_resized = F.interpolate(
                        sam_logits_matched.unsqueeze(1),
                        size=(target_res, target_res),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    
                    # Align dtypes to prevent precision/autocast mismatches
                    sam_logits_matched_resized = sam_logits_matched_resized.to(dtype=student_mask_logits_resized.dtype)
                    
                    # L_mask (KL Divergence on Bernoulli soft probabilities)
                    # FIX: clamp logits before sigmoid to prevent log(0) -> NaN
                    if self.kd_cfg.losses.mask_kd.enabled:
                        sam_clamped = torch.clamp(sam_logits_matched_resized / self.temperature, -15.0, 15.0)
                        stu_clamped = torch.clamp(student_mask_logits_resized / self.temperature, -15.0, 15.0)
                        q = torch.sigmoid(sam_clamped)
                        p_log = F.logsigmoid(stu_clamped)
                        inv_q = 1.0 - q
                        inv_p_log = F.logsigmoid(-stu_clamped)
                        
                        kl = q * (torch.log(q + 1e-8) - p_log) + inv_q * (torch.log(inv_q + 1e-8) - inv_p_log)
                        
                        # Foreground-Dilated / Region-Focused Mask-KL (if enabled)
                        use_focused = getattr(self.kd_cfg.losses.mask_kd, "focused", False) or getattr(self.kd_cfg.losses.mask_kd, "foreground_dilated", False)
                        if use_focused:
                            fg_core = (q > 0.35).float().unsqueeze(1)
                            fg_dilated = F.max_pool2d(fg_core, kernel_size=9, stride=1, padding=4).squeeze(1)
                            fg_core = fg_core.squeeze(1)
                            weight_map = torch.where(fg_core > 0, 1.0, torch.where(fg_dilated > 0, 0.5, 0.05))
                            kl_weighted = (kl * weight_map).sum(dim=(-1, -2)) / (weight_map.sum(dim=(-1, -2)) + 1e-6)
                            loss_mask_kd = loss_mask_kd + kl_weighted.mean() * (self.temperature ** 2)
                        else:
                            loss_mask_kd = loss_mask_kd + kl.mean() * (self.temperature ** 2)
                        loss_mask_kd_count += 1

                    # L_affinity (Spatial Pixel Affinity / Directional Gradient KD)
                    affinity_cfg = getattr(self.kd_cfg.losses, "affinity", None)
                    if affinity_cfg and getattr(affinity_cfg, "enabled", False):
                        p_stu = torch.sigmoid(stu_clamped)
                        d_stu_x = p_stu[:, :, 1:] - p_stu[:, :, :-1]
                        d_stu_y = p_stu[:, 1:, :] - p_stu[:, :-1, :]
                        d_tea_x = q[:, :, 1:] - q[:, :, :-1]
                        d_tea_y = q[:, 1:, :] - q[:, :-1, :]
                        loss_aff = F.mse_loss(d_stu_x, d_tea_x.detach()) + F.mse_loss(d_stu_y, d_tea_y.detach())
                        loss_affinity = loss_affinity + loss_aff
                        loss_affinity_count += 1

                    # L_boundary (Per-instance matched boundary weighted loss)
                    if self.kd_cfg.losses.boundary.enabled:
                        sam_soft = torch.sigmoid(sam_logits_matched_resized / self.temperature)
                        bw = (1.0 - torch.abs(sam_soft - 0.5) * 2).detach()
                        stu_clamped_raw = torch.clamp(student_mask_logits_resized, -30.0, 30.0)
                        bce = F.binary_cross_entropy_with_logits(
                            stu_clamped_raw, sam_soft.detach(), reduction="none"
                        )
                        loss_boundary = loss_boundary + (bce * bw).mean()
                        loss_boundary_count += 1

            if self.kd_cfg.losses.mask_kd.enabled and loss_mask_kd_count > 0:
                kd_losses["mask_kd"] = (loss_mask_kd / loss_mask_kd_count) * self.kd_cfg.losses.mask_kd.weight

            affinity_cfg = getattr(self.kd_cfg.losses, "affinity", None)
            if affinity_cfg and getattr(affinity_cfg, "enabled", False) and loss_affinity_count > 0:
                kd_losses["affinity"] = (loss_affinity / loss_affinity_count) * float(getattr(affinity_cfg, "weight", 1.0))
                
            if self.kd_cfg.losses.boundary.enabled and loss_boundary_count > 0:
                kd_losses["boundary"] = (loss_boundary / loss_boundary_count) * self.kd_cfg.losses.boundary.weight

            # 2. Compute L_feature (Scale-matched alignment)
            if self.kd_cfg.losses.feature.enabled:
                loss_feat = torch.tensor(0.0, device=self.device)
                layers_to_monitor = self.kd_cfg.losses.feature.layers if hasattr(self.kd_cfg.losses.feature, "layers") else [2, 5, 8]
                feat_count = 0
                
                for idx in layers_to_monitor:
                    feat_key = f"layer_{idx}"
                    if feat_key in self.student_features and feat_key in self.proj_layers:
                        sf = self.student_features[feat_key]
                        proj = self.proj_layers[feat_key]
                        sf_proj = proj(sf)
                        
                        feature_h = sf.shape[2]
                        stride = self.args.imgsz // feature_h

                        # Map layer stride directly to SAM feature keys & channels (architecture independent)
                        if stride <= 4:
                            target_key = "feat1"
                            out_channels = 64
                        else:
                            target_key = "image_embed"
                            out_channels = 256
                        
                        # Stack SAM features for the batch
                        tf_list = []
                        for img_path in self._current_paths:
                            stem = Path(img_path).stem
                            if stem in self._sam_features:
                                tf_list.append(self._sam_features[stem][target_key])
                            else:
                                tf_list.append(torch.zeros((1, out_channels, feature_h, feature_h), device=self.device))
                                
                        tf_batch = torch.cat(tf_list, dim=0).to(dtype=sf_proj.dtype)
                        
                        # Resize SAM feature spatially to match student feature
                        if sf_proj.shape[2:] != tf_batch.shape[2:]:
                            tf_batch = F.interpolate(tf_batch, size=sf_proj.shape[2:], mode="bilinear", align_corners=False)
                        
                        # FIX: normalize per-layer MSE so scale doesn't grow with channel dim
                        loss_feat = loss_feat + F.mse_loss(sf_proj, tf_batch.detach())
                        feat_count += 1
                    else:
                        if feat_key not in self.student_features and not self._no_logits_warned:
                            print(f"[KD] Warning: Hook feature {feat_key} not found in student_features. "
                                  f"Forward hooks might not be triggering. Skipping feature KD.")
                            self._no_logits_warned = True
                
                # FIX: average across layers so total feature loss is ~1 layer's worth, not 3×
                if feat_count > 0:
                    loss_feat = loss_feat / feat_count
                kd_losses["feature"] = loss_feat * self.kd_cfg.losses.feature.weight

            # Logging demonstration on first pass
            if not self._kd_logged and kd_losses:
                log_strs = [f"{k}: {float(v):.6f}" for k, v in kd_losses.items()]
                print(f"[KD] ✓ KD losses computed: {', '.join(log_strs)}")
                self._kd_logged = True

        except Exception as e:
            if not self._kd_logged:
                print(f"[KD] Warning: Error computing KD loss: {e} — skipping KD this batch")
                import traceback
                traceback.print_exc()
                self._kd_logged = True

        return kd_losses