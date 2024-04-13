from pydantic import BaseModel

class YoloConfig(BaseModel):
    S: int = 7
    C: int = 5
    B: int = 2
    image_size: int = 448
    lr:float = 3e-6
    weight_decay:float = 0.0
    epochs:int = 1000
    iou_threshold:float = 0.3
    threshold:float = 0.5
    momentum:float = 0.9
    weight_decay:float = 5e-4
    box_format:str = "midpoint"
    annotations_files_path:str = "/home/adriano/Documents/datasets/VOCFULL/VOCdevkit/VOC2012/YOLO"

