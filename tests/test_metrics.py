from __future__ import annotations

import torch

from icassp_topogate.metrics.segmentation import confusion_from_logits, metrics_from_confusion


def test_perfect_metrics():
    target = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)
    logits = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
    m = metrics_from_confusion(confusion_from_logits(logits, target))
    assert m["dice"] > 0.999
    assert m["miou"] > 0.999


if __name__ == "__main__":
    test_perfect_metrics()
    print("metric test passed.")
