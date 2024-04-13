from src.core.config import YoloConfig
from src.core.helpers import mAp, non_max_suppression
from src.datasets.dataset import CarDataset
import torchvision.transforms as transforms
import torch
import random

from src.models.loss import YoloLoss
from src.models.yolo import YoloV1

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# YOLO config
config = YoloConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

train_images_path = "/home/adriano/Documents/datasets/Cars Detection/train/images"
val_images_path = "/home/adriano/Documents/datasets/Cars Detection/train/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

train_dataset = CarDataset(train_images_path, val_images_path, transform=transform)
idx = random.randint(0, len(train_dataset))

image, labels = train_dataset[idx]
image = image.to(device=device).unsqueeze(0)
labels = labels.to(device=device).unsqueeze(0)

# 2 - Model
model = YoloV1()
model.to(device=device)
model.compile()
criterion = YoloLoss()

with torch.no_grad():
    predictions = model(image) 


def convert_cellboxes(predictions, conf: YoloConfig, is_target=True, device=device):

    batch_size = predictions.shape[0]
    
    if is_target:
        target_bboxe = predictions[..., conf.C+1:conf.C+5]
        return target_bboxe

    # for predictions by model
    pred_1, pred_2 = torch.split(predictions, (conf.C + conf.B * 5), dim=3)

    # get bboxes
    pred_1_bboxe = pred_1[..., conf.C+1:conf.C+5]
    pred_2_bboxe = pred_2[..., conf.C+1:conf.C+5]

    # get the scores from the headers 
    scores = torch.cat(
                (pred_1[..., conf.C].unsqueeze(0), pred_2[..., conf.C].unsqueeze(0)), dim=0
            )
    
    # get the best predictions
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = pred_1_bboxe * (1 - best_box) + best_box * pred_2_bboxe
    # get all indicies
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1).to(device=device)
    # get coords and create bboxe
    x = 1 / conf.S * (best_boxes[..., :1] + cell_indices)
    y = 1 / conf.S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / conf.S * best_boxes[..., 2:4]
    
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    # classes predictions
    all_classes_predicted = torch.cat((pred_1[..., :conf.C], pred_2[..., :conf.C]), dim=3) 
    predicted_class = all_classes_predicted[..., :conf.C].argmax(-1).unsqueeze(-1)

    # get the best confidence
    best_confidence = torch.max(pred_1[..., conf.C], pred_2[..., conf.C]).unsqueeze(
            -1
        )
    
    # all results
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out, conf: YoloConfig, is_target=True, device=device):
    converted_pred = convert_cellboxes(out, conf, is_target, device).reshape(out.shape[0], conf.S * conf.S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(conf.S * conf.S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

true_bboxes = cellboxes_to_boxes(labels, config)
preds_bboxes = cellboxes_to_boxes(predictions, config, False, device)


batch_size = predictions.shape[0]

all_pred_boxes = []
all_true_boxes = []

train_idx = 0

for idx in range(batch_size):
    nms_boxes = non_max_suppression(
        preds_bboxes[idx],
        iou_threshold=config.iou_threshold,
        threshold=config.threshold,
        box_format=config.box_format,
    )
    for nms_box in nms_boxes:
        all_pred_boxes.append([train_idx] + nms_box)

    for box in true_bboxes[idx]:
        # many will get converted to 0 pred
        if box[1] > config.threshold:
            all_true_boxes.append([train_idx] + box)

loss = criterion(predictions, labels)
metric_object = mAp(all_pred_boxes, all_true_boxes,box_format="midpoint")

#evaluation_loss = total_loss / len(dataloader)
evaluation_metric = metric_object

print(evaluation_metric)
