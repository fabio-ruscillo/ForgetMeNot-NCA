import torch.nn.functional as F


def ce_dice_loss(logits, target, smooth = 1e-5):
    """
    Computes the combined Cross-Entropy minus the Dice loss

    Args:
        logits (Tensor): Predicted logits of shape (B, C, H, W).
        target (Tensor): Ground-truth labels of shape (B, H, W).
        smooth (float): Smoothing factor for numerical stability.

    Returns:
        tuple [Tensor, Tensor, Tensor]: Total loss, mean cross-entropy, mean Dice score.
    """

    logits = logits.float()
    ce = F.cross_entropy(logits, target)

    # Dice compares predicted probabilities against binary masks for each class.
    probabilities = F.softmax(logits, dim = 1)
    target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (probabilities * target_one_hot).sum(dim = (2, 3))
    total = probabilities.sum(dim = (2, 3)) + target_one_hot.sum(dim = (2, 3))

    # Average per-class Dice scores to give each class equal weight.
    dice = (2 * intersection + smooth) / (total + smooth)
    dice = dice.mean()

    loss = ce - dice

    return loss, ce.detach(), dice.detach()
