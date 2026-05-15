import random
import os
import shutil
import torch
import numpy as np
import cv2

from PIL import Image
import util.vcpi_util as vcpi_util
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image


class RandomSmoothNoise(torch.nn.Module):
    """
    Aproximação rápida de Perlin Noise aconselhada pelo GEMINI (O perlin noise real era muito lento).
    Gera ruído de baixa frequência (manchas suaves) para simular sujidade ou sombras.
    """
    def __init__(self, noise_level=0.15, grid_size=4, p=0.5):
        super().__init__()
        self.noise_level = noise_level
        self.grid_size = grid_size # Matriz inicial (valores menores = manchas maiores)
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img
        
        # Converter PIL para Tensor para operações matemáticas rápidas
        tensor_img = to_tensor(img)
        c, h, w = tensor_img.shape
        
        # 1. Gerar ruído aleatório numa grelha minúscula (ex: 3x4x4) [-1, 1]
        low_res_noise = torch.rand(c, self.grid_size, self.grid_size) * 2 - 1
        
        # 2. Esticar a grelha para o tamanho da imagem usando interpolação bicúbica (cria o efeito Perlin)
        noise = torch.nn.functional.interpolate(
            low_res_noise.unsqueeze(0), 
            size=(h, w), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        # 3. Adicionar o ruído à imagem e garantir que os pixeis ficam entre 0 e 1
        noisy_img = tensor_img + noise * self.noise_level
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        
        # Voltar a PIL Image para o teu pipeline de gravação
        return to_pil_image(noisy_img)

def show_random_samples(dataset, classes, title="Dataset Samples", num_rows=3, num_cols=5):
    """
    Visualizes a grid of random samples from a given dataset.
    What: Picks random indices and displays the images with their labels.
    Why: To visually inspect the data and ensure loading/transforms are correct.
    """
    indices = random.sample(range(len(dataset)), num_rows * num_cols)
    images = [dataset[i][0] for i in indices]
    targets = [dataset[i][1] for i in indices]
    
    print(title)
    vcpi_util.show_images(num_rows, num_cols, images, targets, classes)

def create_train_val_split(origin_path="datasets/origin_train_images", train_path="datasets/train_images", val_path="datasets/val_images", split_ratio=0.2):
    """
    Creates a sequence-aware validation split from the origin_train_images.\n
    Ensures that frames from the same sequence are not shared between train and val.\n
    Stores the Validation imagea in val_images/.\n
    Stores the Training images in train_images/.\n
    If a split already exists, removes the old one and creates a new one.
    """
    
    print(f"Creating sequence-aware split from {origin_path}")
    
    # 1. Clear existing train and val folders
    for path in [train_path, val_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # 2. Extracts the existing classes (e.g: 00000, 00001, ...)
    classes = sorted([d for d in os.listdir(origin_path) if os.path.isdir(os.path.join(origin_path, d))])
    
    for cls in classes:
        # 3. Create a folder in train and val dataset for each class
        cls_origin_path = os.path.join(origin_path, cls)
        cls_train_path = os.path.join(train_path, cls)
        cls_val_path = os.path.join(val_path, cls)
        
        os.makedirs(cls_train_path)
        os.makedirs(cls_val_path)
        
        # 4. Get all the images for a class
        files = [f for f in os.listdir(cls_origin_path) if f.endswith('.ppm')]
        if not files:
            continue

        # 5. Get the id of all sequences in the class
        all_sequences = sorted(list(set([f.split('_')[0] for f in files])))
        
        # 6. Shuffle the IDs for a random selection
        random.shuffle(all_sequences)
        
        # 7. Get the sequence IDs selected to be on the validation dataset using the given split_ratio 
        num_val = int(len(all_sequences) * split_ratio)
        val_sequences = set(all_sequences[:num_val])
        
        for f in files:
            seq_id = f.split('_')[0]
            if seq_id in val_sequences:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_val_path, f))
            else:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_train_path, f))
                    
    print(f"Split complete. Train and Val folders created in {os.path.dirname(train_path)} with a split_ration of {split_ratio}")


def balance_dataset(src_path="datasets/train_images", dest_path="datasets/train_balanced", target_count=500, options=None):
    """
    Balances the dataset by augmenting minority classes using torchvision.transforms.v2.
    """
    # Corrigido para os defaults otimizados que falámos
    if options is None:
        options = {
            'rotate': True,
            'affine': True,
            'color': True,
            'perspective': True,
            'blur': True,
            'sharpness': False, # Desligado (inútil/prejudicial)
            'grayscale': False, # Desligado (destrói semântica do sinal)
            'noise': True,      # Ativado para usar o SmoothNoise
        }

    print(f"Balancing dataset from {src_path} to {dest_path}")
    print(f"Target count: {target_count} images per class")
    print(f"Augmentation options: {options}")

    if os.path.exists(dest_path):
        print(f"Destination folder {dest_path} already exists. Removing it to start fresh...")
        shutil.rmtree(dest_path)
    
    os.makedirs(dest_path)

    # Definir transformações
    aug_list = []
    if options.get('rotate'):
        aug_list.append(v2.RandomRotation(degrees=15))
    if options.get('affine'):
        aug_list.append(v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
    if options.get('color'):
        aug_list.append(v2.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2))
    if options.get('perspective'):
        aug_list.append(v2.RandomPerspective(distortion_scale=0.2, p=0.5))
    if options.get('blur'):
        aug_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3))
    if options.get('sharpness'):
        aug_list.append(v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3))
    if options.get('grayscale'):
        aug_list.append(v2.RandomGrayscale(p=0.3))
    if options.get('noise'):
        # Adiciona a nossa aproximação de Perlin com 50% de probabilidade
        aug_list.append(RandomSmoothNoise(noise_level=0.15, p=0.5))

    aug_transform = v2.Compose(aug_list)

    classes = sorted([d for d in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, d))])

    for cls in classes:
        src_cls_path = os.path.join(src_path, cls)
        dest_cls_path = os.path.join(dest_path, cls)
        os.makedirs(dest_cls_path)

        files = [f for f in os.listdir(src_cls_path) if f.endswith('.ppm') or f.endswith('.png')]
        num_files = len(files)
        
        # 1. Copiar ficheiros originais
        for f in files:
            shutil.copy(os.path.join(src_cls_path, f), os.path.join(dest_cls_path, f))

        # 2. Augmentar se necessário
        if num_files < target_count:
            num_to_augment = target_count - num_files
            for i in range(num_to_augment):
                original_file = random.choice(files)
                img = Image.open(os.path.join(src_cls_path, original_file)).convert('RGB') # Evita erros de canais se a imagem for RGBA ou grelha

                # Aplicar transformações
                new_img = aug_transform(img)

                # Guardar imagem sintética
                new_filename = f"aug_{i}_{original_file.replace('.ppm', '.png')}"
                new_img.save(os.path.join(dest_cls_path, new_filename))

    print("Balancing complete.")

class CLAHE_Transform:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    of the LAB color space, preserving hue and saturation.
    Particularly effective for traffic signs captured under uneven lighting or shadows.
    """
    def __init__(self, clip_limit=2.0, tile_size=(4, 4)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)


class GammaCorrection:
    """
    Randomly adjusts image gamma to simulate varying lighting conditions.
    gamma < 1 brightens dark images; gamma > 1 darkens bright ones.
    """
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        img_np = np.array(img) / 255.0
        img_np = np.power(img_np, gamma)
        return Image.fromarray((img_np * 255).astype(np.uint8))


def get_mean_std(dataset):
    """
    Computes the per-channel mean and standard deviation of a dataset.
    Used to normalize images to zero mean and unit variance before training.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    for imgs, _ in loader:
        for c in range(3):
            mean[c] += imgs[:, c, :, :].mean()
            std[c]  += imgs[:, c, :, :].std()
    mean /= len(loader)
    std  /= len(loader)
    return mean, std


if __name__ == "__main__":
    SRC = 'datasets/train_images'
    DEST = 'datasets/train_balanced'
    balance_dataset(SRC, DEST, target_count=2000)
