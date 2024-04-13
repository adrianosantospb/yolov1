import os
import torch
from torch.utils.data import Dataset
from src.core.config import YoloConfig
import numpy as np
from PIL import Image
from src.core.helpers import letterbox, resize_bboxes

class VOCDataset(Dataset):
    def __init__(self, images_files_path, transform, cache_mode='train'):
        self.conf = YoloConfig()
        self.transform = transform
        self.cache_dir = 'cache'

        # Criar diretório de cache se não existir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Determinar o nome dos arquivos de cache com base no modo (treino ou validação)
        cache_images_filename = f'images_{cache_mode}.npy'
        cache_labels_filename = f'labels_{cache_mode}.npy'

        # Caminhos para os arquivos de cache
        self.cache_images_path = os.path.join(self.cache_dir, cache_images_filename)
        self.cache_labels_path = os.path.join(self.cache_dir, cache_labels_filename)

        if os.path.exists(self.cache_images_path) and os.path.exists(self.cache_labels_path):
            # Carregar dados do cache
            self.images = np.load(self.cache_images_path, allow_pickle=True)
            self.labels = np.load(self.cache_labels_path, allow_pickle=True)
        else:
            images_final = []
            labels_final = []
            with open(images_files_path, 'r') as i:
                self.images = sorted([line.rstrip() for line in i])
                for image_path in self.images:
                    
                    new_image_path = image_path.replace("/JPEGImages/", "/ImageYOLOSets/")
                    new_label_paht = image_path.replace(".jpg", ".txt").replace("/JPEGImages/", "/YOLO/")
                    
                    if os.path.exists(new_image_path) and os.path.exists(new_label_paht): 
                        images_final.append(new_image_path)
                        labels_final.append(new_label_paht)

            self.labels = sorted(labels_final)
            self.images = images_final
            
            # Salvar em cache
            np.save(self.cache_images_path, self.images)
            np.save(self.cache_labels_path, self.labels)

        assert len(self.labels) == len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Carregar imagem
        image = Image.open(self.images[idx])
        W, H = image.size
        
        # letterbox
        image = letterbox(image, self.conf.image_size)

        # Carregar rótulos e caixas delimitadoras
        label_path = self.labels[idx]
        target = np.loadtxt(label_path, dtype=float, delimiter=" ", ndmin=2)
        
        boxes = target[:,1:]
        classes = target[:,0]

        # Redimensiona os bboxes 
        boxes = resize_bboxes(boxes,(W, H), (448, 448)) 

        # Aplica transformação de dados
        image, boxes = self.transform(image, boxes)

        # Converter para tensores do PyTorch
        image = torch.FloatTensor(image)
        boxes = torch.FloatTensor(boxes)

        # Criar matriz de rótulos
        label_matrix = torch.zeros((self.conf.S, self.conf.S, self.conf.C + 5 * self.conf.B))

        for idx, box in enumerate(boxes):
            # Coordenadas da caixa
            x, y, w, h = box.tolist()
            cls = int(classes[idx])

            # Correspondências da grade
            i, j = min(int(self.conf.S * y), self.conf.S - 1), min(int(self.conf.S * x), self.conf.S - 1)
            y_cel, x_cel = self.conf.S * y - int(self.conf.S * y), self.conf.S * x - int(self.conf.S * x)
            x_cel, y_cel = x_cel % 1, y_cel % 1
            h_cel, w_cel = h * self.conf.S, w * self.conf.S

            # Preencher a matriz de rótulos
            label_matrix[i, j, 20] = 1
            label_matrix[i, j, 21:25] = torch.tensor([x_cel, y_cel, w_cel, h_cel])
            label_matrix[i, j, cls] = 1

        return image, label_matrix

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes