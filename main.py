# all
import torch
from tqdm import tqdm

# Dataset
import torchvision.transforms as transforms
from src.datasets.dataset import Compose, CarDataset

# Model
from src.models.yolo import YoloV1
from src.core.config import YoloConfig
import torch.optim as optim
from src.models.loss import YoloLoss
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn

import gc
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()
gc.collect()

# YOLO config
config = YoloConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1 - Dataset
train_transforms = Compose([transforms.Resize((448,448)), transforms.ToTensor()])
val_test_transforms = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

train_images_path = "~/Documents/datasets/Cars Detection/train/images"
train_labels_path = "~/Documents/datasets/Cars Detection/train/labels"

val_images_path = "~/Documents/datasets/Cars Detection/valid/images"
val_labels_path = "~/Documents/datasets/Cars Detection/valid/labels"

num_worker = 4 * int(torch.cuda.device_count())
batch_size = 8

train_dataset = CarDataset(train_images_path, train_labels_path, transform=train_transforms, cache_mode="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_worker, pin_memory=True)

val_dataset = CarDataset(val_images_path, val_labels_path, transform=val_test_transforms, cache_mode="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker, pin_memory=True)

print("Training datset {} Val dataset {} ".format(len(train_dataset), len(val_dataset)))

# 2 - Model
model = YoloV1()
model.to(device=device)
model.compile()

criterion = YoloLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = OneCycleLR(optimizer, max_lr=config.lr, epochs = config.epochs, steps_per_epoch = 2*(len(train_dataloader)), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')

# 3 - training
from src.core.helpers import cellboxes_to_boxes, mAp, non_max_suppression


def train(train_loader, model, criterion, optimizer, device):
    torch.cuda.empty_cache()
    
    model.train()
    total_loss = 0.0
    
    for _ , (image, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = image.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        predictions = model(image)
        loss = criterion(predictions, labels)
        total_loss += loss.item()               
        
        loss.backward()

        # Clip gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        # adjust learning rate
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
            true_bboxes = cellboxes_to_boxes(labels, config)
            bboxes = cellboxes_to_boxes(y_preds, config, False, device)

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
           

    evaluation_metric = mAp(all_pred_boxes, all_true_boxes,box_format="midpoint")
    evaluation_loss = total_loss / len(dataloader)

    return evaluation_loss, evaluation_metric


if __name__ == "__main__":    
    try:
        best_metric = 0

        for epoch in range(config.epochs):

            print("\nEpoch {}".format(1 + epoch))
            
            train_mean_loss = train(train_loader=train_dataloader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer, device=device)
            
            print("Train mean loss: {}".format(train_mean_loss))
            
            evaluation_loss, evaluation_metric = evaluating(model=model, dataloader=val_dataloader, criterion=criterion, device=device)
            print("Val mean loss: {} mAp: {}".format(evaluation_loss, evaluation_metric))

            if best_metric < evaluation_metric:
                torch.save(model.state_dict(), "./weights/best.pt")
                best_metric = evaluation_metric

        torch.cuda.empty_cache()
            
    except Exception as ex:
        print(ex)

    del model
    torch.cuda.empty_cache()
    gc.collect()
