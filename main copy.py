# all
import torch
from tqdm import tqdm

# Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.datasets.dataset import VOCDataset

# Model
from src.models.yolo import YoloV1
from src.core.config import YoloConfig
import torch.optim as optim
from src.models.loss import YoloLoss
from torch.optim.lr_scheduler import OneCycleLR
import cv2

import gc
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
gc.collect()

# YOLO config
config = YoloConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1 - Dataset
scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(config.image_size * scale)),
        A.PadIfNeeded(
            min_height=int(config.image_size * scale),
            min_width=int(config.image_size * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=config.image_size, height=config.image_size),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.6, label_fields=[],),
)

val_test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size, min_width=config.image_size, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.6, label_fields=[]),
)

train_images_path = "/home/adriano/Documents/datasets/VOCFULL/old_txt_files/train.txt"
val_images_path = "/home/adriano/Documents/datasets/VOCFULL/old_txt_files/test.txt"

num_worker = 4 * int(torch.cuda.device_count())
batch_size = 8

train_dataset = VOCDataset(train_images_path, transform=train_transforms, cache_mode="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_worker, pin_memory=True)

val_dataset = VOCDataset(val_images_path, transform=val_test_transforms, cache_mode="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker, pin_memory=True)

print(len(train_dataset), len(val_dataset))

# 2 - Model
model = YoloV1()
model.to(device=device)
model.compile()

criterion = YoloLoss()
#optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = OneCycleLR(optimizer, max_lr=config.lr, epochs = config.epochs, steps_per_epoch = 2*(len(train_dataloader)), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')

# 3 - training
from src.core.helpers import cellboxes_to_boxes, mAp, non_max_suppression


def train(train_loader, model, criterion, optimizer, device):
    torch.cuda.empty_cache()

    total_loss = 0.0
    
    for _ , (image, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        for param in model.parameters():
            param.grad = None

        image = image.to(device)
        labels = labels.to(device)
        
        predictions = model(image)
        loss = criterion(predictions, labels)
        total_loss += loss.item()               
        
        loss.backward()

        optimizer.step()

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()

    return total_loss / len(train_loader)

def evaluating(model, dataloader, criterion, device):
    torch.cuda.empty_cache()

    model.eval()
    total_loss = 0.0
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_preds = model(inputs)

            batch_size = inputs.shape[0]
            true_bboxes = cellboxes_to_boxes(labels)
            bboxes = cellboxes_to_boxes(y_preds)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
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

            train_idx += 1


            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()
           

    metric_object = mAp(all_pred_boxes, all_true_boxes,box_format="midpoint")

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object
    return evaluation_loss, evaluation_metric

try:
    for epoch in range(config.epochs):

        print("\nEpoch {}".format(1 + epoch))
        
        train_mean_loss = train(train_loader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer, device=device)
        
        print("Train mean loss: {}".format(train_mean_loss))
        
        evaluation_loss, evaluation_metric = evaluating(model=model, dataloader=val_dataloader, criterion=criterion, device=device)
        print("Val mean loss: {} mAp: {}".format(evaluation_loss, evaluation_metric))
 
        torch.cuda.empty_cache()
        
except Exception as ex:
    print(ex)

del model
torch.cuda.empty_cache()
gc.collect()