#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    src_path = Path("/home/shahin/distill/final_notebooks/04_run_pixel_affinity_kd.ipynb")
    dst_path = Path("/home/shahin/distill/final_notebooks/09_run_focal_mask_kd.ipynb")

    with open(src_path) as f:
        nb = json.load(f)

    # Update Cell 0 markdown
    nb['cells'][0]['source'] = [
        "# 🚀 Crack-Distill: Research Variant: Focal Mask-KL (Seed 42)\n",
        "Penalizes uncertain sub-pixel boundary predictions by applying focal weighting $(1 - p_{\\text{student}})^\\gamma$ ($\\gamma=2.0$) to Bernoulli Mask-KL.\n",
        "\n",
        "### Recipe Specifications:\n",
        "* **Student Model**: YOLOv11n-seg (2.84M parameters, 10.2 GFLOPs)\n",
        "* **Teacher Model**: SAM 2 Large (`sam2_hiera_large.pt`, 224M parameters)\n",
        "* **Prompt Type**: Bounding Box Only (offline pre-computed soft logits)\n",
        "* **Temperature**: $\\tau = 3.7769$, Mask KD Weight: $W = 0.9612$, Focal $\\gamma = 2.0$\n",
        "* **Head Freezing**: Disabled (Full end-to-end training)\n",
        "* **Precision**: FP32 (`amp: false`) for 100% loss stability\n",
        "* **Random Seed**: `42`\n",
        "* **Automated Evaluation**: Evaluates in-domain cropped val and out-of-distribution uncropped val upon completion.\n"
    ]

    # Update Cell 6 (kd_trainer.py) to include focal weighting in KDSegmentationTrainer
    trainer_code = ''.join(nb['cells'][6]['source'])
    old_kl_block = """                        kl = q * (torch.log(q + 1e-8) - p_log) + inv_q * (torch.log(inv_q + 1e-8) - inv_p_log)
                        
                        # Foreground-Dilated / Region-Focused Mask-KL (if enabled)"""

    new_kl_block = """                        kl = q * (torch.log(q + 1e-8) - p_log) + inv_q * (torch.log(inv_q + 1e-8) - inv_p_log)
                        
                        # Focal Mask-KL weighting: (1 - p_student)^gamma (if enabled)
                        if getattr(self.kd_cfg.losses.mask_kd, "focal", False):
                            p_stu = torch.sigmoid(stu_clamped)
                            gamma = float(getattr(self.kd_cfg.losses.mask_kd, "focal_gamma", 2.0))
                            focal_w = (1.0 - p_stu).detach() ** gamma
                            kl = focal_w * kl

                        # Foreground-Dilated / Region-Focused Mask-KL (if enabled)"""

    assert old_kl_block in trainer_code, 'old_kl_block not found in trainer code'
    trainer_code_fixed = trainer_code.replace(old_kl_block, new_kl_block)
    nb['cells'][6]['source'] = [line + '\n' for line in trainer_code_fixed.split('\n')][:-1]

    # Update Cell 12 (Step 3: Training)
    nb['cells'][12]['source'] = [
        "# ── Step 3: Run Focal Mask-KL Training (exp_focal_mask_kd_g2_T3.7769_W0.9612_seed42_150ep) ──\n",
        "import sys\n",
        "sys.path.insert(0, \".\")\n",
        "from pathlib import Path\n",
        "from distillation.kd_trainer import KDSegmentationTrainer\n",
        "from utils.config_loader import load_config, override_config\n",
        "\n",
        "cfg = load_config(\"configs/config.yaml\")\n",
        "EXPERIMENT_NAME = \"exp_focal_mask_kd_g2_T3.7769_W0.9612_seed42_150ep\"\n",
        "\n",
        "overrides = {\n",
        "    \"distillation.enabled\": True,\n",
        "    \"distillation.temperature\": 3.7769,\n",
        "    \"distillation.progressive.enabled\": False,\n",
        "    \"distillation.losses.task.weight\": 1.0,\n",
        "    \"distillation.losses.mask_kd.enabled\": True,\n",
        "    \"distillation.losses.mask_kd.weight\": 0.9612,\n",
        "    \"distillation.losses.mask_kd.focal\": True,        # Focal Mask-KL active\n",
        "    \"distillation.losses.mask_kd.focal_gamma\": 2.0,   # gamma = 2.0\n",
        "    \"distillation.losses.mask_kd.focused\": False,\n",
        "    \"distillation.losses.affinity.enabled\": False,\n",
        "    \"distillation.losses.feature.enabled\": False,\n",
        "    \"distillation.losses.boundary.enabled\": False\n",
        "}\n",
        "overrides[\"project.name\"] = \"crack_distill\"\n",
        "overrides[\"project.experiment\"] = EXPERIMENT_NAME\n",
        "overrides[\"project.seed\"] = 42\n",
        "overrides[\"data.datasets\"] = [{\"name\": \"crack500\", \"path\": \"data/datasets/crack500_yolo\", \"format\": \"yolo\"}]\n",
        "overrides[\"teacher.logits_dir\"] = \"data/teacher_logits_box/\"\n",
        "overrides[\"train.epochs\"] = 150\n",
        "overrides[\"train.amp\"] = False\n",
        "\n",
        "cfg = override_config(cfg, overrides)\n",
        "\n",
        "print(f\"=== Starting Run: {EXPERIMENT_NAME} ===\")\n",
        "print(\"Config in effect: Focal Mask-KL (gamma=2.0), Seed=42, Epochs=150, AMP=False, Freezing=False\")\n",
        "\n",
        "trainer = KDSegmentationTrainer(cfg)\n",
        "trainer.train()\n",
        "print(\"✓ Training completed successfully!\")\n"
    ]

    # Update Cell 13 (Step 4: Evaluation)
    nb['cells'][13]['source'] = [
        "# ── Step 4: Validate Best Checkpoint & Compare vs. All Baselines ──\n",
        "import glob, json, os\n",
        "from pathlib import Path\n",
        "from ultralytics import YOLO\n",
        "\n",
        "KNOWN_BASELINES = {\n",
        "    \"No-KD Baseline (EXP-10)\": {\"Mask mAP50\": 0.5400, \"Box mAP50\": 0.5970, \"OOD mAP50\": 0.0848},\n",
        "    \"01_seed42 (Locked Baseline)\": {\"Mask mAP50\": 0.5424, \"Box mAP50\": 0.5976, \"OOD mAP50\": 0.0848},\n",
        "    \"03_dilated (Best OOD)\": {\"Mask mAP50\": 0.5387, \"Box mAP50\": 0.5819, \"OOD mAP50\": 0.1007},\n",
        "    \"04_affinity (Best In-Domain)\": {\"Mask mAP50\": 0.5569, \"Box mAP50\": 0.5973, \"OOD mAP50\": 0.0842},\n",
        "    \"05_multiscale (Best Box mAP)\": {\"Mask mAP50\": 0.5485, \"Box mAP50\": 0.6001, \"OOD mAP50\": 0.0872},\n",
        "    \"06_layerkd (Best mAP50-95)\": {\"Mask mAP50\": 0.5538, \"Box mAP50\": 0.5947, \"OOD mAP50\": 0.0799},\n",
        "}\n",
        "\n",
        "best_pt = glob.glob(f\"runs/**/{EXPERIMENT_NAME}*/weights/best.pt\", recursive=True)\n",
        "assert best_pt, f\"No checkpoint found for {EXPERIMENT_NAME}! Check run directory.\"\n",
        "\n",
        "print(f\"Evaluating best checkpoint: {best_pt[0]}\")\n",
        "model = YOLO(best_pt[0])\n",
        "\n",
        "# 1. Validate on Crack500 In-Domain Val Set\n",
        "print(\"\\n--- In-Domain Cropped Validation ---\")\n",
        "val_metrics = model.val(data=\"data/datasets/crack500_yolo/dataset.yaml\", split=\"val\", verbose=True)\n",
        "mask_map50 = float(val_metrics.seg.map50)\n",
        "mask_map95 = float(val_metrics.seg.map)\n",
        "box_map50  = float(val_metrics.box.map50)\n",
        "box_map95  = float(val_metrics.box.map)\n",
        "\n",
        "results = {\n",
        "    \"experiment\": EXPERIMENT_NAME,\n",
        "    \"seed\": 42,\n",
        "    \"checkpoint\": best_pt[0],\n",
        "    \"metrics_indomain\": {\n",
        "        \"mask_mAP50\": mask_map50,\n",
        "        \"mask_mAP50_95\": mask_map95,\n",
        "        \"box_mAP50\": box_map50,\n",
        "        \"box_mAP50_95\": box_map95,\n",
        "        \"mask_precision\": float(val_metrics.seg.p[0]) if hasattr(val_metrics.seg, \"p\") and len(val_metrics.seg.p) > 0 else float(val_metrics.seg.mp),\n",
        "        \"mask_recall\": float(val_metrics.seg.r[0]) if hasattr(val_metrics.seg, \"r\") and len(val_metrics.seg.r) > 0 else float(val_metrics.seg.mr)\n",
        "    }\n",
        "}\n",
        "\n",
        "# 2. Validate on Crack500 Uncropped (OOD) Set\n",
        "ood_mask_map50 = None\n",
        "uncropped_yaml = Path(\"data/datasets/crack500_uncropped_yolo/dataset.yaml\")\n",
        "if uncropped_yaml.exists():\n",
        "    print(\"\\n--- Out-of-Distribution (Uncropped) Validation ---\")\n",
        "    ood_metrics = model.val(data=str(uncropped_yaml), split=\"val\", verbose=True)\n",
        "    ood_mask_map50 = float(ood_metrics.seg.map50)\n",
        "    results[\"metrics_ood\"] = {\n",
        "        \"ood_mask_mAP50\": ood_mask_map50,\n",
        "        \"ood_mask_mAP50_95\": float(ood_metrics.seg.map),\n",
        "        \"ood_box_mAP50\": float(ood_metrics.box.map50),\n",
        "        \"ood_box_mAP50_95\": float(ood_metrics.box.map)\n",
        "    }\n",
        "else:\n",
        "    print(\"Warning: Uncropped dataset.yaml not found — skipping OOD evaluation.\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*72)\n",
        "print(f\"📊 HEAD-TO-HEAD COMPARISON vs. EXISTING BASELINES\")\n",
        "print(\"=\"*72)\n",
        "print(f\"  {'Model':<32} {'In-Domain Mask':>16} {'OOD Mask':>12} {'Box mAP50':>12}\")\n",
        "print(\"-\"*72)\n",
        "for name, vals in KNOWN_BASELINES.items():\n",
        "    print(f\"  {name:<32} {vals['Mask mAP50']:>16.4f} {vals['OOD mAP50']:>12.4f} {vals['Box mAP50']:>12.4f}\")\n",
        "print(\"-\"*72)\n",
        "ood_str = f\"{ood_mask_map50:.4f}\" if ood_mask_map50 is not None else \"N/A\"\n",
        "print(f\"  {'⭐ FOCAL MASK-KL (gamma=2.0)':<32} {mask_map50:>16.4f} {ood_str:>12} {box_map50:>12.4f}  ← THIS RUN\")\n",
        "print(\"=\"*72)\n",
        "\n",
        "delta_base = mask_map50 - 0.5400\n",
        "delta_aff  = mask_map50 - 0.5569\n",
        "print(f\"  vs. Baseline (0.5400)      : {delta_base:+.4f} ({delta_base/0.5400*100:+.1f}%)\")\n",
        "print(f\"  vs. Best Affinity (0.5569) : {delta_aff:+.4f} ({delta_aff/0.5569*100:+.1f}%)\")\n",
        "if ood_mask_map50 is not None:\n",
        "    delta_ood_base = ood_mask_map50 - 0.0848\n",
        "    delta_ood_dil  = ood_mask_map50 - 0.1007\n",
        "    print(f\"  OOD vs. Baseline (0.0848)  : {delta_ood_base:+.4f} ({delta_ood_base/0.0848*100:+.1f}%)\")\n",
        "    print(f\"  OOD vs. 03_dilated (0.1007): {delta_ood_dil:+.4f} ({delta_ood_dil/0.1007*100:+.1f}%)\")\n",
        "print()\n",
        "\n",
        "out_file = Path(f\"/kaggle/working/results/{EXPERIMENT_NAME}.json\")\n",
        "out_file.parent.mkdir(parents=True, exist_ok=True)\n",
        "with open(out_file, \"w\") as f:\n",
        "    json.dump(results, f, indent=2)\n",
        "print(f\"Saved structured summary to {out_file}\")\n"
    ]

    with open(dst_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated {dst_path} (size: {dst_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
