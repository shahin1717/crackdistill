import json
from pathlib import Path

import numpy as np

from distillation.kd_trainer import KDScaleCalibrator
from scripts.make_pseudo_labels import teacher_maps

ROOT = Path(__file__).resolve().parent.parent


def _line(x1, y1, x2, y2):
    return f"0 {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def test_teacher_maps_one_map_per_label_line_and_neighbours_are_background():
    low = np.full((2, 256, 256), -8.0, np.float32)
    low[0, 20:40, 20:120] = 6.0      # crack A
    low[1, 180:200, 100:220] = 4.0   # crack B
    lines = [_line(20 / 256, 20 / 256, 120 / 256, 40 / 256), _line(100 / 256, 180 / 256, 220 / 256, 200 / 256)]
    maps = teacher_maps(low, lines)
    assert maps.shape == (2, 256, 256) and maps.dtype == np.float16
    assert maps[0, 30, 60] == 6 and maps[0, 190, 150] == -4     # A keeps itself, B is background for A
    assert maps[1, 190, 150] == 4 and maps[1, 30, 60] == -6
    assert maps[0, 100, 100] == -8 and maps[1, 100, 100] == -8  # background stays the teacher's background


def test_teacher_maps_single_instance_is_the_union_of_the_box_logits():
    low = np.random.default_rng(0).normal(size=(3, 256, 256)).astype(np.float32)
    maps = teacher_maps(low, [_line(0.1, 0.1, 0.9, 0.9)])
    assert maps.shape == (1, 256, 256)
    np.testing.assert_allclose(maps[0], low.max(0).astype(np.float16))


def test_calibrator_freezes_the_weight_at_the_target_fraction():
    cal = KDScaleCalibrator({"mask_kd": 0.25, "boundary": None}, steps=3)
    assert cal("boundary", 5.0, 10.0) == 1.0                    # no auto_frac: configured weight is used as is
    assert cal("mask_kd", 0.0, 10.0) == 0.0                     # an empty term is skipped, not counted
    for term in (2.0, 4.0, 6.0):                                # mean term 4, mean task 8
        scale = cal("mask_kd", term, 8.0)
    assert abs(scale - 0.25 * 8.0 / 4.0) < 1e-9
    assert cal("mask_kd", 100.0, 1.0) == scale                  # frozen after `steps` batches


def _cells(name):
    nb = json.loads((ROOT / "loss_notebooks" / f"{name}_seed0.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"]]


def _step(cells, n):
    return next(c for c in cells if c.startswith(f"# ── Step {n}"))


def test_loss_notebooks_start_from_r2_and_never_use_attached_logits():
    for name in ("L0_control", "L1_mask_kd", "L2_boundary", "L3_feature_kd"):
        cells = _cells(name)
        run = [c for c in cells if not c.startswith("%%writefile")]
        assert not any("generate_teacher_logits.py --prompt-type" in c for c in run), name
        assert not any("os.symlink(src_f, dst_f)" in c for c in run), name
        step3 = _step(cells, 3)
        assert 'overrides["student.backbone"] = str(STUDENT_INIT)' in step3
        assert "crack500_pseudo/dataset.yaml" in step3 and "'lr0': 0.0002" in step3
        assert "'train.epochs': 40" in step3 and "'train.d4_aug': True" in step3
        assert "sam_labels_p25_seed0" in _step(cells, "2b")


def test_loss_notebooks_differ_only_in_the_kd_term():
    on = {name: {t for t in ("mask_kd", "boundary", "feature")
                 if f"'distillation.losses.{t}.enabled': True" in _step(_cells(name), 3)}
          for name in ("L0_control", "L1_mask_kd", "L2_boundary", "L3_feature_kd")}
    assert on == {"L0_control": set(), "L1_mask_kd": {"mask_kd"}, "L2_boundary": {"boundary"},
                  "L3_feature_kd": {"feature"}}
    assert "'distillation.enabled': False" in _step(_cells("L0_control"), 3)
    assert "--kd-out" not in _step(_cells("L0_control"), "2b")
    assert "--features" in _step(_cells("L3_feature_kd"), "2b") and "--features" not in _step(_cells("L1_mask_kd"), "2b")
