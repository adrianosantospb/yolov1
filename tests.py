from src.core.config import YoloConfig
from src.datasets.dataset import CarDataset
import torchvision.transforms as transforms

import random


conf = YoloConfig()

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

image, matrix = train_dataset[idx] 
print(matrix)