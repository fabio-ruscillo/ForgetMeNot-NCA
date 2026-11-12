import math
import pytest
import torch
from nca_memory.loss import ce_dice_loss


def test_perfect_prediction_approaches_minus_one():
    """Check the ideal loss with absent classes to ensure that in the limit case of perfect prediction everything works fine."""
    target = torch.zeros((1, 10, 10), dtype=torch.long)

    logits = torch.zeros((1, 10, 10, 10))
    logits[:, 0] = 40.0

    loss, _, _ = ce_dice_loss(logits, target)

    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_uniform_prediction_matches_expected_loss():
    """Check the formula against an analytically computed result to validate the mathematical implementation."""
    smooth = 1e-5
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    logits = torch.zeros((1, 10, 10, 10))

    loss, _, _ = ce_dice_loss(logits, target, smooth=smooth)

    present_dice = (20.0 + smooth) / (110.0 + smooth)
    absent_dice = smooth / (10.0 + smooth)
    expected_dice = (present_dice + 9 * absent_dice) / 10
    expected_loss = math.log(10) - expected_dice

    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)


def test_wrong_prediction_has_higher_loss():
    """Verify that the loss is higher when the model predicts worngly, to ensure proper penalization."""
    target = torch.zeros((1, 10, 10), dtype=torch.long)

    correct = torch.zeros((1, 10, 10, 10))
    correct[:, 0] = 40.0

    wrong = torch.zeros_like(correct)
    wrong[:, 1] = 40.0

    correct_loss, _, _ = ce_dice_loss(correct, target)
    wrong_loss, _, _ = ce_dice_loss(wrong, target)

    assert wrong_loss.item() > correct_loss.item()


def test_batch_loss_equals_mean_sample_loss():
    """Verify batch averaging to ensure that each sample contribute queally to the loss. """
    target = torch.zeros((2, 10, 10), dtype=torch.long)
    logits = torch.zeros((2, 10, 10, 10))

    logits[0, 0] = 5.0
    logits[1, 1] = 5.0

    batch_loss, _, _ = ce_dice_loss(logits, target)

    individual_losses = [
        ce_dice_loss(logits[i:i+1], target[i:i+1])[0]
        for i in range(2)
    ]

    expected_loss = torch.stack(individual_losses).mean()

    assert batch_loss.item() == pytest.approx(expected_loss.item(), abs=1e-6)


def test_backward_produces_finite_gradients():
    """Check gradient finiteness with absent classes to prevent numerical instability during training."""
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    logits = torch.zeros((1, 10, 10, 10), requires_grad=True)

    loss, _, _ = ce_dice_loss(logits, target)
    gradient = torch.autograd.grad(loss, logits)[0]

    assert torch.isfinite(gradient).all()


def test_backward_produces_nonzero_gradients():
    """Check for nonzero gradients with absent classes to detect potential vanishing gradient issues."""
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    logits = torch.zeros((1, 10, 10, 10), requires_grad=True)

    loss, _, _ = ce_dice_loss(logits, target)
    gradient = torch.autograd.grad(loss, logits)[0]

    assert torch.count_nonzero(gradient) > 0
