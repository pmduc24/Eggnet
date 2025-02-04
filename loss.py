import torch
import torch.nn as nn
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation import MeanIoU
from torchmetrics import MetricCollection

class Unified(nn.Module):
    """Unified activation function module."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        self.softplus = nn.Softplus(beta=-1.0)
        self.lambda_param = nn.Parameter(lambda_param)
        self.kappa_param = nn.Parameter(kappa_param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        l = torch.clamp(self.lambda_param, min=0.0001)
        p = torch.exp((1 / l) * self.softplus((self.kappa_param * input) - torch.log(l)))
        return p * input # for AGLU simply return p*input

def dice_loss(pred, target, smooth=1e-6):
    # Dự đoán (sau softmax) và nhãn one-hot
    pred = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
    target = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])  # [batch_size, H, W, num_classes]
    target = target.permute(0, 3, 1, 2)  # [batch_size, num_classes, H, W]

    # Tính Dice Loss
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 2.0 * intersection / (union + smooth)
    return 1 - dice.mean()  # Dice Loss = 1 - Dice Score

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_dice=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # Tính CrossEntropyLoss
        ce = self.ce_loss(logits, target)

        # Tính Dice Loss
        dice = dice_loss(logits, target)

        # Kết hợp hai hàm loss
        return (1 - self.weight_dice) * ce + self.weight_dice * dice

class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()
        self.dice = GeneralizedDiceScore()

    def forward(self, logits, target):
        return 1 - self.dice(logits, target)

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_dice=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.weight_dice = GeneralizedDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # Tính CrossEntropyLoss
        ce = self.ce_loss(logits, target)

        # Tính Dice Loss
        dice = dice_loss(logits, target)

        # Kết hợp hai hàm loss
        return (1 - self.weight_dice) * ce + self.weight_dice * dice