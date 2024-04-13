import torch
import torch.nn as nn
from src.core.config import YoloConfig
from src.core.helpers import iou

class YoloLoss(nn.Module):
    """
    Calculate the loss for YOLO (v1) model.
    """

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.conf = YoloConfig()

        self.mse = nn.MSELoss(reduction="sum")

        # These are from the YOLO paper, indicating how much we should
        # weigh the loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # Compute loss using combined predictions
        loss = self.compute_loss(predictions, targets)

        return loss

    def compute_loss(self, predictions, target):
        # Verify if the number of samples is consistent between predictions and targets
        assert predictions.shape[0] == target.shape[0], "The number of samples in predictions and targets must be the same"

        pred_1, pred_2 = torch.split(predictions, (self.conf.C + self.conf.B * 5), dim=3)
        real_target_bboxes = target[..., self.conf.C+1:self.conf.C+5]

        # Calculate IoU for the two predicted bounding boxes with target bbox (torch.Size([2, 8, 7, 7, 1]))
        pred_1_bboxe = pred_1[..., self.conf.C+1:self.conf.C+5]
        pred_2_bboxe = pred_2[..., self.conf.C+1:self.conf.C+5]

        iou_b1 = iou(pred_1_bboxe, real_target_bboxes)
        iou_b2 = iou(pred_1_bboxe, real_target_bboxes)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.conf.C].unsqueeze(3)  # in paper this is Iobj_i

        #   FOR BOX COORDINATES    #
        box_predictions = exists_box * (
            (
                bestbox * pred_2_bboxe
                + (1 - bestbox) * pred_1_bboxe
            )
        )

        box_targets = exists_box * real_target_bboxes

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        #   FOR OBJECT LOSS    #
        pred_box = (
            bestbox * pred_2[..., self.conf.C:self.conf.C+1] + (1 - bestbox) * pred_1[..., self.conf.C:self.conf.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.conf.C:self.conf.C+1]),
        )

        #   FOR NO OBJECT LOSS    #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * pred_1[..., self.conf.C:self.conf.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.conf.C:self.conf.C+1], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * pred_2[..., self.conf.C:self.conf.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.conf.C:self.conf.C+1], start_dim=1)
        )
        
        #   FOR CLASS LOSS   #
        class_loss = self.mse(
            torch.flatten(exists_box * pred_1[..., :self.conf.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.conf.C], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        
        return loss
