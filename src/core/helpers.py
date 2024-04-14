import torch
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps
from collections import Counter

from src.core.config import YoloConfig

# Label map
voc_labels = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')

label_map = {k: v for v, k in enumerate(voc_labels)}
#label_map['background'] = 0

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return boxes, labels, difficulties

# Convertion from corner points to relative mid points
def from_x1y1x2y2_to_xywh(bboxes, H, W):
    dw = 1./W
    dh = 1./H
    
    w = round((bboxes[2] - bboxes[0]) * dw, 2) # w
    h = round((bboxes[3] - bboxes[1]) * dh, 2) # h
    x = round(((bboxes[0] + bboxes[2]) / 2) * dw, 2) # x
    y = round(((bboxes[1] + bboxes[3]) / 2) * dh, 2) # y
    
    x = abs(max(0.01, min(1, x)))
    y = abs(max(0.01, min(1, y)))
    h = abs(max(0.01, min(1, h)))
    w = abs(max(0.01, min(1, w)))

    return [x, y, w, h]

# Convertion from relative mid points to corner points
def from_xywh_to_x1y1x2y2(bboxes, H, W):
    x1 = round((bboxes[0] - bboxes[2] / 2) * W) # x1
    y1 = round((bboxes[1] - bboxes[3] / 2 ) * H) # y1
    x2 = round((bboxes[0] + bboxes[2] / 2 ) * W) # x2
    y2 = round((bboxes[1] + bboxes[3] / 2 ) * H) # y2
    
    return (x1, y1, x2, y2)

def convert_cellboxes(predictions, conf: YoloConfig, is_target=True, device="cuda"):

    batch_size = predictions.shape[0]
    
    if is_target:
        _, class_indices = torch.max(predictions[..., :conf.C], dim=-1)
        # Extrair a confiança
        confidence = predictions[..., conf.C]
        # Extrair as coordenadas do bounding box
        bbox_coordinates = predictions[..., conf.C + 1: conf.C + 5]
        # Concatenar as informações extraídas em um tensor de saída
        target_bboxe = torch.cat((class_indices.unsqueeze(-1), confidence.unsqueeze(-1), bbox_coordinates), dim=-1)
        
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


def cellboxes_to_boxes(out, conf: YoloConfig, is_target=True, device="cuda"):
    converted_pred = convert_cellboxes(out, conf, is_target, device).reshape(out.shape[0], conf.S * conf.S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(conf.S * conf.S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

'''
Non max suppression

'''
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

'''
mAp - Mean Average Precision (mAP) 
https://www.youtube.com/watch?v=FppOzcDvaDI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=4
'''
def iou(predictions, targets, box_format="midpoint"):
    """
    Calculate Intersection over Union (IoU) for predicted and target bounding boxes.

    Args:
        predictions (tensor): Predicted bounding boxes, shape (N, 4), where N is the number of boxes.
        targets (tensor): Target bounding boxes, shape (N, 4).
        box_format (str): Format of the bounding boxes, either 'midpoint' or 'corners'.

    Returns:
        tensor: IoU for each pair of predicted and target bounding boxes, shape (N,).
    """
    
    if box_format == "midpoint":
        # Get coordinates of the predicted bounding boxes
        p_x1 = predictions[..., 0:1] - predictions[..., 2:3] / 2  # left
        p_y1 = predictions[..., 1:2] - predictions[..., 3:4] / 2  # top
        p_x2 = predictions[..., 0:1] + predictions[..., 2:3] / 2  # right
        p_y2 = predictions[..., 1:2] + predictions[..., 3:4] / 2  # bottom

        # Get coordinates of the target bounding boxes
        t_x1 = targets[..., 0:1] - targets[..., 2:3] / 2  # left
        t_y1 = targets[..., 1:2] - targets[..., 3:4] / 2  # top
        t_x2 = targets[..., 0:1] + targets[..., 2:3] / 2  # right
        t_y2 = targets[..., 1:2] + targets[..., 3:4] / 2  # bottom
        
    elif box_format == "corners":
        # Get coordinates of the predicted bounding boxes
        p_x1, p_y1, p_x2, p_y2 = predictions[..., 0:1], predictions[..., 1:2], predictions[..., 2:3], predictions[..., 3:4]

        # Get coordinates of the target bounding boxes
        t_x1, t_y1, t_x2, t_y2 = targets[..., 0:1], targets[..., 1:2], targets[..., 2:3], targets[..., 3:4]

    # Get intersection coordinates
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    # Compute intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute area of prediction and target boxes
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)

    # Compute union area
    union_area = p_area + t_area - inter_area

    # Add epsilon for numerical stability
    epsilon = 1e-6

    # Compute IoU
    iou = inter_area / (union_area + epsilon)

    return iou

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

    
    # Verifica se houve detecções positivas em todas as classes para evitar divisão por zero
    if len(average_precisions) > 0:
        recall = sum(recalls) / len(recalls) if recalls.numel() > 0 else 0.001
        precision = sum(precisions) / len(precisions) if precisions.numel() > 0 else 0.001
        mAP_value = sum(average_precisions) / len(average_precisions)
    else:
        recall = torch.tensor(0)
        precision = torch.tensor(0)
        mAP_value = torch.tensor(0)
    
    return mAP_value, recall, precision


def resize_bboxes(bboxes, old_size, new_size):
    """
    Redimensiona as coordenadas das caixas delimitadoras (bboxes) de acordo com a nova dimensão da imagem.

    Args:
    - bboxes (list): Uma lista de coordenadas das caixas delimitadoras no formato YOLO (x, y, w, h).
    - old_size (tuple): As dimensões antigas da imagem original (width, height).
    - new_size (int): A nova dimensão para a imagem redimensionada (o mesmo valor para largura e altura).

    Returns:
    - list: Uma lista contendo as novas coordenadas das caixas delimitadoras redimensionadas no formato YOLO (x, y, w, h).
    """
    old_w, old_h = old_size
    new_w, new_h = new_size

    resized_bboxes = []

    for bbox in bboxes:
        x, y, w, h = bbox

        # Calcula os fatores de escala para largura e altura
        scale_w = new_w / old_w
        scale_h = new_h / old_h

        # Redimensiona as coordenadas da caixa delimitadora
        x_new = x * scale_w
        y_new = y * scale_h
        w_new = w * scale_w
        h_new = h * scale_h

        resized_bboxes.append((x_new, y_new, w_new, h_new))

    return resized_bboxes


def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    bboxes = []
    for mask in Y:
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            bboxes.append(np.zeros(4, dtype=np.float32))
        else:
            y1 = np.min(rows)
            x1 = np.min(cols)
            y2 = np.max(rows)
            x2 = np.max(cols)
            bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return np.array(bboxes)

# Letterbox function
def letterbox(image, desired_size):
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = image.resize((new_size[0], new_size[1]), Image.LANCZOS)
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    return ImageOps.expand(im, padding, fill='black')