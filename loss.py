import torch
import torch.nn as nn
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation import MeanIoU
from torchmetrics import MetricCollection

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