import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import random

class MultiDegradeDataset(Dataset):
    def __init__(self, degrade_data_path, clean_data_path, contrastive=False):
        super().__init__()
        self.degrade_data_path = degrade_data_path
        self.clean_data_path = clean_data_path
        self.degrade_types = ['noisy', 'blurry', 'hazy', 'darkness', 'rainy']
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.contrastive = contrastive
        self._build_dataset()
        self._init_trans()


    def _init_trans(self):
        self.trans_funcs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(480,640),antialias=True),
            # transforms.Normalize(self.mean, self.std)
        ])
        
    def _build_dataset(self):
        self.clean_data_files: list[str] = os.listdir(self.clean_data_path)
        if len(self.clean_data_files) == 0:
            print(f"Empty Dataset!!!:{self.clean_data_path}")
        self.degrade_files: list[str] = os.listdir(self.degrade_data_path)
        if len(self.degrade_files) == 0:
            print(f"Empty Dataset!!!:{self.degrade_data_path}")
        self.dagraed_clean_maping: dict[str,dict[str,str|int]] = {}
        # { "/<degrade_data_path>/123456#1_10_5_1_4.jpg": {
        #         "clean_file_path": "/<clean_data_path>/123456.jpg",
        #         "degrade": {"hazy": 1, "rainy": 10, "blury": 5, "low_light": 1, "noisy": 4}
        #   }
        # }
        self.clean_dagraed_maping: dict[str,list[str]] = {}
        # { "/<clean_file_path>/123456.jpg": [
        #        "/<degrade_data_path>/123456#1_10_5_1_4.jpg",
        #        "/<degrade_data_path>/123456#2_10_5_1_4.jpg",
        #   ]
        # }
        print("Loading Dataset...")
        degrade_types = ['hazy', 'rainy', 'blurry', 'darkness', 'noisy']
        for degrade_file in tqdm(self.degrade_files):
            # degrade_file_file 型如 123456#1_10_5_1_4.jpg
            file_id, degrade_degrees = degrade_file.split('.')[0].split('#')
            degrade_degrees = degrade_degrees.split('_')
            degrade_file_path = os.path.join(self.degrade_data_path, degrade_file)
            clean_file_path = os.path.join(self.clean_data_path, file_id) + '.jpg'
            self.dagraed_clean_maping[degrade_file_path] =  {
                "clean_file_path": clean_file_path,
                "degrade": {
                   degrade_types[i]: degrade_degrees[i] for i in range(5)
                }
            }
            if clean_file_path not in self.clean_dagraed_maping.keys():
                self.clean_dagraed_maping[clean_file_path] = [degrade_file_path]
            else:
                self.clean_dagraed_maping[clean_file_path].append(degrade_file_path)
        print("Loading Dataset Finished!")


    def _trans(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return self.trans_funcs(image)
    
    def _build_degrade_mat(self, degrade_dict):
        return torch.tensor(
            [int(degrade_dict[degrade_type]) for degrade_type in self.degrade_types]
        )
    

    def __len__(self):
        return len(self.degrade_files)

    def __getitem__(self, index):
        degrade_file = self.degrade_files[index]
        degrade_file_path =  os.path.join(self.degrade_data_path, degrade_file)
        clean_file_path = self.dagraed_clean_maping[degrade_file_path]['clean_file_path']
        degrade_image = self._trans(degrade_file_path)
        clean_image = self._trans(clean_file_path)
        if not self.contrastive:
            return clean_image, degrade_image, \
                self._build_degrade_mat(self.dagraed_clean_maping[degrade_file_path]['degrade'])
        else:
            negtive_img_paths = self.clean_dagraed_maping[clean_file_path]
            negtive_imgs = []
            for negtive_img_path in negtive_img_paths:
                negtive_imgs.append(self._trans(negtive_img_path))
            return clean_image, degrade_image, torch.stack(random.choices(negtive_imgs, k=5)), \
                self._build_degrade_mat(self.dagraed_clean_maping[degrade_file_path]['degrade'])

      
        
    



if __name__ == '__main__':
    train_dataset = MultiDegradeDataset(
        '/data/data2/lyh/train2017_degrade',
        '/data/data2/lyh/train2017'
    )
    print(train_dataset[138])
