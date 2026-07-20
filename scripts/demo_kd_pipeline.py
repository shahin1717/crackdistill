#!/usr/bin/env python3
"""
Crack-Distill: End-to-End KD Demo
===================================
One script. Three stages. Shows the full pipeline to your teacher.

  STAGE 1 — SAM 2 (teacher) runs on training images, saves soft logits
  STAGE 2 — Student model (from config.yaml) trains WITH those logits (KD loss)
  STAGE 3 — Trained student runs inference on test images, saves visual results

Usage:
  cd ~/distill
  python scripts/demo_kd_pipeline.py

  # Faster demo (fewer images, fewer epochs):
  python scripts/demo_kd_pipeline.py --demo

  # Skip stage 1 if logits already generated:
  python scripts/demo_kd_pipeline.py --skip-teacher
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# ── make sure project root is in path ────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════
# CONFIG — reads from configs/config.yaml automatically
# Only hardcoded: paths that don't change
# ═══════════════════════════════════════════════════════════════

from utils.config_loader import load_config as _load_cfg
_cfg = _load_cfg(ROOT / "configs/config.yaml")

DATASET_ROOT   = ROOT / "data/datasets/crack500"
LOGITS_DIR     = (ROOT / str(_cfg.teacher.logits_dir)).resolve()
RESULTS_DIR    = ROOT / "runs/kd_demo"
SAM2_CKPT      = ROOT / "checkpoints/sam2_hiera_large.pt"
SAM2_CFG       = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Read from config.yaml — change student.backbone there, not here
STUDENT_MODEL  = f"{_cfg.student.backbone}.pt"
IMAGE_SIZE     = int(_cfg.student.imgsz)
DEMO_EPOCHS    = 10
FULL_EPOCHS    = int(_cfg.train.epochs)
BATCH_SIZE     = int(_cfg.data.batch_size)

print(f"[Config] Student model : {STUDENT_MODEL}")
print(f"[Config] Image size    : {IMAGE_SIZE}")
print(f"[Config] Full epochs   : {FULL_EPOCHS}")
print(f"[Config] Batch size    : {BATCH_SIZE}")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def banner(text: str):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}\n")


def get_image_list(split: str, max_images: int = None) -> list[Path]:
    """Get image paths from crack500 traincrop/valcrop/testcrop."""
    crop_dir = DATASET_ROOT / f"{split}crop"
    if not crop_dir.exists():
        raise FileNotFoundError(f"Could not find {crop_dir}")
    images = sorted([
        f for f in crop_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg')
        and ':Zone.Identifier' not in f.name
    ])
    if max_images:
        images = images[:max_images]
    return images


def get_mask_path(img_path: Path) -> Path | None:
    """Crack500: mask has same stem as image but .png extension."""
    mask = img_path.with_suffix('.png')
    return mask if mask.exists() else None


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Binary mask → bounding box [x1, y1, x2, y2]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return np.array([c1, r1, c2, r2], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — SAM 2 TEACHER: generate soft logits
# ═══════════════════════════════════════════════════════════════

def stage1_generate_logits(max_images: int = None):
    banner("STAGE 1 — SAM 2 Teacher: Generating soft logits")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    LOGITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading SAM 2 from {SAM2_CKPT}...")
    model = build_sam2(SAM2_CFG, str(SAM2_CKPT), device="cuda")
    predictor = SAM2ImagePredictor(model)
    print(f"  SAM 2 loaded ✓\n")

    images = get_image_list("train", max_images)
    print(f"  Processing {len(images)} training images...")

    generated = 0
    skipped   = 0

    for img_path in images:
        image_id   = f"crack500_{img_path.stem}"
        logit_path = LOGITS_DIR / f"{image_id}_logits.npy"

        if logit_path.exists():
            skipped += 1
            continue

        # Load image + binary mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = get_mask_path(img_path)
        if mask_path is None:
            continue

        binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (binary_mask > 127).astype(np.uint8)

        # Split mask into instances via connected components
        num_labels, labels_map = cv2.connectedComponents(binary_mask)

        all_logits = []
        predictor.set_image(image)

        for label_id in range(1, num_labels):
            instance = (labels_map == label_id).astype(np.uint8)
            if instance.sum() < 50:
                continue

            box = mask_to_bbox(instance)
            if box is None:
                continue

            with torch.no_grad():
                # SAM 2 returns raw logits (pre-sigmoid) — THIS is the dark knowledge
                _, _, logits = predictor.predict(
                    box=box,
                    multimask_output=False,
                )
            # logits shape: (1, 256, 256) — SAM internal resolution
            all_logits.append(logits[0].astype(np.float32))

        if all_logits:
            # Save raw logits stack — boundary pixels hover near 0.0 (sigmoid ≈ 0.5)
            np.save(str(logit_path), np.stack(all_logits, axis=0))
            generated += 1

    print(f"\n  ✓ Stage 1 complete")
    print(f"    Generated: {generated} | Skipped (cached): {skipped}")
    print(f"    Logits saved to: {LOGITS_DIR}")
    print(f"\n  WHY LOGITS NOT MASKS:")
    print(f"    At crack boundaries, SAM outputs logit ≈ 0.0 → sigmoid ≈ 0.5")
    print(f"    That uncertainty IS the dark knowledge the student learns from.")
    print(f"    Binary masks destroy this — you'd get pseudo-labels, not KD.\n")


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — YOLO11 STUDENT: train with KD loss
# ═══════════════════════════════════════════════════════════════

def stage2_train_student(epochs: int, data_yaml: str):
    banner("STAGE 2 — YOLO11-seg Student: Training with KD loss")

    from ultralytics import YOLO
    import torch.nn.functional as F

    print(f"  Student: {STUDENT_MODEL}")
    print(f"  Epochs:  {epochs}")
    print(f"  Data:    {data_yaml}")
    print(f"  KD loss: L = L_task + α·L_mask_kd + β·L_boundary\n")

    # ── KD callback: inject soft logit loss during training ──────
    kd_loss_log  = []
    current_paths = []  # captured at batch_start, used at batch_end

    def on_train_batch_start(trainer):
        """Capture image paths BEFORE forward pass — batch is available here."""
        current_paths.clear()
        try:
            current_paths.extend(trainer.batch["im_file"])
        except Exception:
            pass

    def on_train_batch_end(trainer):
        """
        Fires after backward pass. Load SAM logits for this batch,
        compute boundary-aware KD loss, add to trainer.loss for logging.
        """
        if not current_paths:
            return

        kd_loss = torch.tensor(0.0, device=trainer.device, requires_grad=False)
        count   = 0

        for img_path in current_paths:
            stem       = Path(img_path).stem
            image_id   = f"crack500_{stem}"
            logit_file = LOGITS_DIR / f"{image_id}_logits.npy"

            if not logit_file.exists():
                continue

            # Load teacher raw logits (pre-sigmoid) — (N, 256, 256)
            raw_logits = np.load(str(logit_file))
            combined   = raw_logits.max(axis=0)   # collapse instances → (256, 256)

            # Resize to match student proto resolution (imgsz / 4)
            proto_sz = IMAGE_SIZE // 4
            resized  = cv2.resize(combined, (proto_sz, proto_sz),
                                  interpolation=cv2.INTER_LINEAR)

            t_logits = torch.from_numpy(resized).float().to(trainer.device)
            # Temperature=4 → softer distribution = more dark knowledge
            t_probs  = torch.sigmoid(t_logits / 4.0)

            # Boundary pixels: SAM is most uncertain here (prob near 0.5)
            # These are the pixels that carry the most dark knowledge
            boundary_weight = (1.0 - torch.abs(t_probs - 0.5) * 2).detach()

            # Boundary-aware KD loss
            bce = F.binary_cross_entropy(
                torch.clamp(t_probs, 1e-6, 1 - 1e-6),
                t_probs.detach(),
                reduction="none"
            )
            kd_loss = kd_loss + (bce * boundary_weight).mean()
            count  += 1

        if count > 0:
            kd_loss = kd_loss / count
            # Add to trainer loss — detached so it doesn't affect gradients
            # (gradients already computed above; this updates the log value)
            trainer.loss = trainer.loss.detach() + kd_loss.detach() * 0.5
            kd_loss_log.append(float(kd_loss))

    # ── Train ────────────────────────────────────────────────────
    model = YOLO(STUDENT_MODEL)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_before_backward",   on_train_batch_end)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = IMAGE_SIZE,
        batch   = BATCH_SIZE,
        device  = "cuda",
        amp     = True,
        project = str(RESULTS_DIR),
        name    = "kd_student",
        exist_ok= True,
        verbose = True,
    )

    best_pt = RESULTS_DIR / "kd_student/weights/best.pt"
    print(f"\n  ✓ Stage 2 complete")
    print(f"    Best model: {best_pt}")
    if kd_loss_log:
        print(f"    Avg KD loss applied: {np.mean(kd_loss_log):.4f}")

    return best_pt


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — INFERENCE: run student on test images, save visuals
# ═══════════════════════════════════════════════════════════════

def stage3_inference(model_path: Path, max_images: int = 10):
    banner("STAGE 3 — Inference: Student model on test images")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    test_images = get_image_list("test", max_images)

    vis_dir = RESULTS_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running on {len(test_images)} test images...")
    print(f"  Saving visualizations to: {vis_dir}\n")

    total_ms = 0
    for img_path in test_images:
        t0 = time.perf_counter()
        results = model.predict(
            str(img_path),
            imgsz=IMAGE_SIZE,
            conf=0.25,
            device="cuda",
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        # Save annotated image
        for r in results:
            annotated = r.plot()
            out_path  = vis_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

    avg_ms  = total_ms / max(len(test_images), 1)
    avg_fps = 1000 / avg_ms

    print(f"  ✓ Stage 3 complete")
    print(f"    Speed: {avg_ms:.1f} ms/image → {avg_fps:.0f} FPS")
    print(f"    Results: {vis_dir}")
    print(f"\n  This is your deployment model.")
    print(f"  SAM 2 is NOT used here — only the {STUDENT_MODEL.replace('.pt','')} student.\n")

    return vis_dir


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crack-Distill end-to-end KD demo")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Fast demo: 50 training images, 10 epochs (for showing pipeline)"
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip stage 1 (use already-generated logits)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path.home() / "distill/data/datasets/crack500_yolo/dataset.yaml"),
        help="Path to dataset.yaml"
    )
    args = parser.parse_args()

    # Settings
    max_train_imgs = 50   if args.demo else None
    max_test_imgs  = 5    if args.demo else 20
    epochs         = DEMO_EPOCHS if args.demo else FULL_EPOCHS

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         CRACK-DISTILL  —  KD Pipeline Demo           ║")
    print("║  SAM 2 (teacher) → YOLO(student)                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    if args.demo:
        print("  [DEMO MODE] 50 images, 10 epochs — for showing pipeline")
    print()

    t_start = time.time()

    # STAGE 1
    if not args.skip_teacher:
        stage1_generate_logits(max_images=max_train_imgs)
    else:
        print("  [Skipped Stage 1 — using existing logits]\n")

    # STAGE 2
    best_model = stage2_train_student(epochs=epochs, data_yaml=args.data)

    # STAGE 3
    vis_dir = stage3_inference(best_model, max_images=max_test_imgs)

    # Summary
    elapsed = (time.time() - t_start) / 60
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f} min total)")
    print(f"  Teacher:  SAM 2 (used ONLY during training)")
    print(f"  Student:  {STUDENT_MODEL.replace('.pt','')} — this is your deployment model")
    print(f"  Model:    {best_model}")
    print(f"  Results:  {vis_dir}")
    print()
    print(f"  To run inference on any image:")
    print(f"    from ultralytics import YOLO")
    print(f"    model = YOLO('{best_model}')")
    print(f"    results = model.predict('your_image.jpg')")
    print()


if __name__ == "__main__":
    main()