import os
import torch
from torch.utils.data import Dataset
from src.core.config import YoloConfig
import numpy as np
from PIL import Image
from src.core.helpers import letterbox, resize_bboxes
import glob

class CarDataset(Dataset):
    def __init__(self, images_files_path, labels_files_path, transform, cache_mode='train'):
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
            self.images = [os.path.join(images_files_path, filepath) for filepath in glob.glob(images_files_path)]
            self.labels = [os.path.join(labels_files_path, filepath) for filepath in glob.glob(labels_files_path)]
            
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

        print(boxes)

        # Criar matriz de rótulos
        label_matrix = torch.zeros((self.conf.S, self.conf.S, self.conf.C + 5 * self.conf.B))

        for idx, box in enumerate(boxes):
            # Coordenadas da caixa
            x, y, w, h = box.tolist()
            cls = int(classes[idx])

            # Correspondências da grade
            i, j = int(self.conf.S * y), int(self.conf.S * x)
            y_cel, x_cel =  self.conf.S * y - i, self.conf.S * x -j
            h_cel, w_cel = h + self.conf.S, w + self.conf.S

            # Preencher a matriz de rótulos
            label_matrix[i, j, self.conf.C] = 1
            label_matrix[i, j, self.conf.C+1:self.conf.C+5] = torch.tensor([x_cel, y_cel, w_cel, h_cel])
            label_matrix[i, j, cls] = 1

        return image, label_matrix

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes