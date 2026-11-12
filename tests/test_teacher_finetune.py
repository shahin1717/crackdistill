import numpy as np

from scripts.sam2_teacher_finetune import accumulate, instances, new_acc, summarize, thin_mask


def test_instances_skip_small_components_and_box_xyxy():
    gt = np.zeros((40, 60), np.uint8)
    gt[5:15, 10:30] = 1    # 200 px -> kept
    gt[30:33, 50:53] = 1   # 9 px -> dropped (< 50)
    masks, boxes = instances(gt)
    assert len(masks) == 1 and boxes.tolist() == [[10, 5, 29, 14]]


def test_metrics_perfect_and_empty_prediction():
    gt = np.zeros((20, 20), np.uint8)
    gt[8:10, 2:18] = 1     # 2 px wide crack -> all thin
    assert thin_mask(gt).sum() == gt.sum()
    perfect = summarize(accumulate(new_acc(), gt.astype(bool), gt))
    assert perfect["f1"] == 1 and perfect["mean_iou"] == 1 and perfect["thin_recall"] == 1
    empty = summarize(accumulate(new_acc(), np.zeros_like(gt, bool), gt))
    assert empty["recall"] == 0 and empty["f1"] == 0


def test_thick_crack_is_not_thin():
    gt = np.zeros((40, 40), np.uint8)
    gt[10:30, 5:35] = 1
    assert thin_mask(gt)[20, 20] == 0
