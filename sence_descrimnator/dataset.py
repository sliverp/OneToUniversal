import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class MultiDegradeDataset(Dataset):
    def __init__(self, degrade_data_path, clean_data_path):
        super().__init__()
        self.degrade_data_path = degrade_data_path
        self.clean_data_path = clean_data_path
        self._build_dataset()
        self._init_trans()


    def _init_trans(self):
        self.trans_funcs = transforms.Compose([
            transforms.ToTensor()
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
        print("Loading Dataset...")
        degrade_types = ['hazy', 'rainy', 'blury', 'low_light', 'noisy']
        for degrade_file in tqdm(self.degrade_files):
            # degrade_file_file 型如 123456#1_10_5_1_4.jpg
            file_id, degrade_degrees = degrade_file.split('.')[0].split('#')
            degrade_degrees = degrade_degrees.split('_')
            self.dagraed_clean_maping[
                os.path.join(self.degrade_data_path, degrade_file)
            ] =  {
                "clean_file_path": os.path.join(self.clean_data_path, file_id) + '.jpg',
                "degrade":{
                   degrade_types[i]: degrade_degrees[i] for i in range(5)
                }
            }
        print("Loading Dataset Finished!")


    def _trans(self, image_path):
        image = Image.open(image_path)
        return self.trans_funcs(image)
        

    def __len__(self):
        return len(self.degrade_files)

    def __getitem__(self, index):
        degrade_file = self.degrade_files[index]
        degrade_file_path =  os.path.join(self.degrade_data_path, degrade_file)
        clean_file_path = self.dagraed_clean_maping[degrade_file_path]['clean_file_path']
        degrade_image = self._trans(degrade_file_path)
        clean_image = self._trans(clean_file_path)
        return clean_image, degrade_image, self.dagraed_clean_maping[degrade_file_path]['degrade']
    



if __name__ == '__main__':
    train_dataset = MultiDegradeDataset(
        '/data/data2/lyh/train2017_degrade',
        '/data/data2/lyh/train2017'
    )
    print(train_dataset[138])
