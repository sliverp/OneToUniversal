import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from utils_word_embedding import initialize_wordembedding_matrix



class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.type_name = ['clear', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']
        self.mid_dim = 1024
        self.out_dim = 324
        self._setup_word_embedding()


    def _setup_word_embedding(self):
        self.type2idx = {self.type_name[i]: i for i in range(len(self.type_name))}
        self.num_type = len(self.type_name)
        train_type = [self.type2idx[type_i] for type_i in self.type_name]
        self.train_type = torch.LongTensor(train_type).to("cuda" if torch.cuda.is_available() else "cpu")

        wordemb, self.word_dim = initialize_wordembedding_matrix('glove', self.type_name)
        
        self.embedder = torch.nn.Embedding(self.num_type, self.word_dim)
        self.embedder.weight.data.copy_(wordemb)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.word_dim, self.out_dim),
            torch.nn.ReLU(True)
        )
        
class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()


class SenceDescrimnator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()


if __name__ == '__main__':
    sd = SenceDescrimnator()
    