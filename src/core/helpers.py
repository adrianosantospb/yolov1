from typing import Counter
import torch
import xml.etree.ElementTree as ET
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageOps

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

def iou(predictions, targets, box_format="midpoint"):
    
    if box_format == "midpoint":
        # obtem os itens das bboxes preditas
        p_box_x1 = predictions[..., 0:1] - predictions[..., 2:3] / 2 # x1
        p_box_y1 = predictions[..., 1:2] - predictions[..., 3:4] / 2 # y1
        p_box_x2 = predictions[..., 0:1] + predictions[..., 2:3] / 2 # x2
        p_box_y2 = predictions[..., 1:2] + predictions[..., 3:4] / 2 # y2

        # obtem os itens das bboxes preditas
        t_box_x1 = targets[..., 0:1] - targets[..., 2:3] / 2 # x1
        t_box_y1 = targets[..., 1:2] - targets[..., 3:4] / 2 # y1
        t_box_x2 = targets[..., 0:1] + targets[..., 2:3] / 2 # x2
        t_box_y2 = targets[..., 1:2] + targets[..., 3:4] / 2 # y2 
        
    elif box_format == "corners":
        # obtem os itens das bboxes preditas
        p_box_x1 = predictions[..., 0:1] # x1
        p_box_y1 = predictions[..., 1:2] # y1
        p_box_x2 = predictions[..., 2:3] # x2
        p_box_y2 = predictions[..., 3:4] # y2

        # obtem os itens das bboxes preditas
        t_box_x1 = targets[..., 0:1] # x1
        t_box_y1 = targets[..., 1:2] # y1
        t_box_x2 = targets[..., 2:3] # x2
        t_box_y2 = targets[..., 3:4] # y2

    # obtem os maximos valores superiores
    x1 = torch.max(p_box_x1, t_box_x1)
    y1 = torch.max(p_box_y1, t_box_y1)

    # obtem os minimos valores inferiores
    x2 = torch.min(p_box_x2, t_box_x2)
    y2 = torch.min(p_box_y2, t_box_y2)

    # clamp eh uma funcao utilizada para caso nao haja intercesao o valor atribuido sera 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculando a area dos bbox. Utilizaremos a funcao abs para garantir que o valor seja positivo.
    t_box_area = abs((t_box_x2 - t_box_x1) * (t_box_y2 - t_box_y1))
    p_box_area = abs((p_box_x2 - p_box_x1) * (p_box_y2 - p_box_y1))

    return intersection/ (t_box_area + p_box_area - intersection + 1e-6) # adiciona estabilidade numerica...


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

'''
Non max suppression

predictions = list of bbox. Ex: ([[1 , 0.9, x1, y1, x2, y2], [1 , 0.3, x1, y1, x2, y2],[1 , 0.61, x1, y1, x2, y2]])
                classe, prob, bbox
iou_threshold = filtro
box_format = formato do bbox midpoint ou corners

https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c

'''
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
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

def mAp(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou_value = iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

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



def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))            

            if not objects:
                continue

            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))