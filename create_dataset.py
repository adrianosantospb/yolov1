import xml.etree.ElementTree as ET
import os
import shutil

# Mapeamento de nomes de classe para índices
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

class_to_index = {label: str(i) for i, label in enumerate(voc_labels)}

# Função para ler e processar um arquivo de anotação XML VOC
def processar_arquivo_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    annotations = []
    for obj in root.findall('object'):
        obj_class = obj.find('name').text
        obj_index = class_to_index.get(obj_class, '-1')  # Se a classe não estiver no mapeamento, atribui -1
                
        bbox = obj.find('bndbox')
        xmin = round(float(bbox.find('xmin').text) -1)
        ymin = round(float(bbox.find('ymin').text) -1)
        xmax = round(float(bbox.find('xmax').text) -1)
        ymax = round(float(bbox.find('ymax').text) -1)

        # Normalizar as coordenadas das caixas delimitadoras
        x_center = (xmin + xmax) / (2.0 * width)
        y_center = (ymin + ymax) / (2.0 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        annotations.append((obj_index, x_center, y_center, box_width, box_height))

    return annotations

# Diretório contendo os arquivos XML de anotação
xml_dir = '/home/adriano/Documents/datasets/VOCFULL/VOCdevkit/VOC2007/Annotations'

# Diretório contendo as imagens
image_dir = '/home/adriano/Documents/datasets/VOCFULL/VOCdevkit/VOC2007/JPEGImages'

# Diretório de saída para os arquivos YOLO
output_dir = '/home/adriano/Documents/datasets/VOCFULL/VOCdevkit/VOC2007/YOLO'

# Iterar sobre os arquivos XML
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_dir, xml_file)
        annotation = processar_arquivo_xml(xml_path)

        if len(annotation) > 0:            
            # Obter o nome da imagem correspondente
            image_filename = os.path.splitext(xml_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_filename)

            new_image_path = image_path.replace("JPEGImages", "ImageYOLOSets") 
            
            shutil.copy(image_path, new_image_path)

            # Salvar a anotação no formato Darknet YOLO
            output_filename = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
            with open(output_filename, 'w') as f:
                for obj_index, x_center, y_center, box_width, box_height in annotation:
                    f.write(f"{obj_index} {x_center} {y_center} {box_width} {box_height}\n")