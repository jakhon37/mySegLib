"""Training must hand the losses raw logits: softmax first puts a floor under the loss."""
import ast
from pathlib import Path

import pytest
import torch

from src.losses.combined_loss import CombinedLoss, DiceLoss
from src.losses.focal_loss import FocalLoss

SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.mark.parametrize("script", ["train.py", "train_dp.py"])
def test_training_scripts_build_the_model_without_an_activation(script):
    tree = ast.parse((SCRIPTS / script).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "create_model"
    ]
    assert calls
    for call in calls:
        activation = next(k.value for k in call.keywords if k.arg == "activation")
        assert isinstance(activation, ast.Constant) and activation.value is None


def test_a_perfect_prediction_only_costs_nothing_as_logits():
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[0, 20:30, 20:30] = 1
    logits = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float() * 30
    loss = CombinedLoss(
        FocalLoss(gamma=2.0), DiceLoss(mode="multiclass", from_logits=True), alpha=0.5
    )
    assert loss(logits, target).item() < 0.001
    assert loss(logits.softmax(1), target).item() > 0.25
