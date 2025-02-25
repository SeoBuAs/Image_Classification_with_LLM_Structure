import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        label_map = {
            'AGC': 0,
            'Dysplasia': 1,
            'EGC': 2,
            'Normal': 3
        }
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                if folder in label_map:
                    label = label_map[folder]
                    for image_name in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(label)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
