from collections import Counter
from src.core.config import YoloConfig
from src.core.helpers import cellboxes_to_boxes, iou, non_max_suppression
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


batch_size = image.shape[0]

true_bboxes = cellboxes_to_boxes(labels, config)
preds_bboxes = cellboxes_to_boxes(predictions, config, False, device)
all_pred_boxes = []
all_true_boxes = []

train_idx = 0

for idx in range(batch_size):
    nms_boxes = non_max_suppression(
        preds_bboxes[idx],
        iou_threshold=config.iou_threshold,
        threshold= config.threshold,
        box_format=config.box_format,
    )

    for nms_box in nms_boxes:
        all_pred_boxes.append([train_idx] + nms_box)
    
    for box in true_bboxes[idx]:
        # many will get converted to 0 pred
        if box[1] > config.threshold:
            all_true_boxes.append([train_idx] + box)
    
    train_idx += 1


def mAP(predictions, targets, iou_threshold=0.5, box_format="midpoint", num_classes=5):

    assert type(predictions) == list
    assert type(targets) == list

    # list storing all AP for respective classes
    average_precisions = []

    epsilon = 1e-6 #https://nhigham.com/2020/08/04/what-is-numerical-stability/

    for c in range(num_classes): # varre todas as possiveis classes
        detections = []
        ground_truths = []

        for det in predictions: # agrupa as deteccoes por classe
            if det[1] == c:
                detections.append(det)
        
        for gt in targets: # agrupa as anotacoes reais por classe
            if gt[1] == c:
                ground_truths.append(gt)
        
        # Ex: {classe_a: n, classe_b: m, ...}
        class_dict = Counter([gt[0] for gt in ground_truths])

        # Ex: {classe_a: torch.tensor([0,0,0]), classe_b: torch.tensor([0,0,0,0,0]), ...}
        for i, item in class_dict.items():
            class_dict[i] = torch.zeros(item)

        # Ordena a lista do maior para o menor
        detections.sort(key=lambda x:x[2], reverse=True)
        
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        if len(detections) == 0:
            continue

        for det_idx, det in enumerate(detections):
            # obtem apenas os ground_truths que correspondem a classe especifica detectada
            ground_truths_img = [bbox for bbox in ground_truths if bbox[0]==det[0]]

            best_iou = 0

            for idx, gt in enumerate(ground_truths_img):
                _iou = iou(torch.tensor(det[3:]), torch.tensor(gt[3:]), box_format=box_format)
                
                if _iou > best_iou:
                    best_iou = _iou
                    best_gt_idx = idx

        for det_idx, det in enumerate(detections):
            # obtem apenas os ground_truths que correspondem a classe especifica detectada
            ground_truths_img = [bbox for bbox in ground_truths if bbox[0]==det[0]]

            best_iou = 0

            for idx, gt in enumerate(ground_truths_img):
                _iou = iou(torch.tensor(det[3:]), torch.tensor(gt[3:]), box_format=box_format)
                
                if _iou > best_iou:
                    best_iou = _iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if class_dict[det[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[det_idx] = 1
                    class_dict[det[0]][best_gt_idx] = 1
                else:
                    FP[det_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[det_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        
        if recalls.numel() == 0 or precisions.numel() == 0:
            # Retorna valores padrão se os tensores estiverem vazios
            average_precisions.append(torch.tensor(0))
        else:
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

    mAP_value = sum(average_precisions) / len(average_precisions)
    
    # Verifica se houve detecções positivas em todas as classes para evitar divisão por zero
    if len(average_precisions) > 0:
        recall = sum(recalls) / len(recalls)
        precision = sum(precisions) / len(precisions)
    else:
        recall = torch.tensor(0)
        precision = torch.tensor(0)
                
    return mAP_value, recall, precision

map, recall, precision = mAP(all_pred_boxes, all_true_boxes, box_format="midpoint", num_classes=config.C)
print(map, recall, precision)