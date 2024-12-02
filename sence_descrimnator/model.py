import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from tqdm import tqdm
from PIL import Image
from .dataset import MultiDegradeDataset
from .utils_word_embedding import initialize_wordembedding_matrix



class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.type_name = ['noisy', 'blurry', 'hazy', 'darkness', 'rainy']
        self.out_dim = 324
        self._setup_word_embedding()


    def _setup_word_embedding(self):
        self.type2idx = {self.type_name[i]: i for i in range(len(self.type_name))}
        self.num_type = len(self.type_name)
        train_type = [self.type2idx[type_i] for type_i in self.type_name]
        self.train_type = torch.LongTensor(train_type).to("cuda" if torch.cuda.is_available() else "cpu")

        self.wordemb, self.word_dim = initialize_wordembedding_matrix('glove', self.type_name)
        self.word_mat = []
        for degrade_type in self.type_name:
            self.word_mat.append(self.wordemb[degrade_type])
        self.word_mat = torch.stack(self.word_mat)
        if torch.cuda.is_available():
            self.word_mat = self.word_mat.cuda()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.word_dim, self.out_dim),
            torch.nn.ReLU(True)
        )

    def forward(self, degrade_mat):
        # degrade_mat = torch.tensor([gaussian_noise, motion_blur, haze, low_light, rainy])
        # print(f"self.word_mat:{self.word_mat.shape}")
        # print(f"degrade_mat:{degrade_mat.shape}")
        b, _ = degrade_mat.shape
        word_code = (self.word_mat) * degrade_mat.reshape(b,5,1)
        # print(word_code)
        word_code = word_code.sum(dim=1)
        # print(word_code)
        word_code = (word_code - word_code.mean()) / word_code.std()
        # print(f"word_code:{word_code.shape}")
        # word_code = self.mlp(word_code)
        # print(word_code)
        return word_code
        
        

class ImageEncoder(torch.nn.Module):
    class Backbone(torch.nn.Module):
        def __init__(self, backbone='resnet18'):
            super(ImageEncoder.Backbone, self).__init__()
            if backbone == 'resnet18':
                resnet = torchvision.models.resnet.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet50':
                resnet = torchvision.models.resnet.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet101':
                resnet = torchvision.models.resnet.resnet101(
                    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)

            self.block0 = torch.nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            )
            self.block1 = resnet.layer1
            self.block2 = resnet.layer2
            self.block3 = resnet.layer3
            self.block4 = resnet.layer4

        def forward(self, x, returned=[4]):
            blocks = [self.block0(x)]
            blocks.append(self.block1(blocks[-1]))
            blocks.append(self.block2(blocks[-1]))
            blocks.append(self.block3(blocks[-1]))
            blocks.append(self.block4(blocks[-1]))
            out = [blocks[i] for i in returned]
            return out

    def __init__(self):
        super().__init__()
        self.extractor_name = 'resnet101'
        self.mid_dim = 1024
        self.feat_dim = 2048
        self.out_dim =  300
        self.drop_rate =  0.35
        self._setup_image_embedding()

    def forward(self, images):
        img = self.feat_extractor(images)[0]
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze(3).squeeze(2)
        img = self.img_final(img)
        
        return img

    def _setup_image_embedding(self):
        # image embedding
        self.feat_extractor = self.Backbone(self.extractor_name)
        self.feat_extractor.requires_grad_(False)

        img_emb_modules = [
            torch.nn.Conv2d(self.feat_dim, self.mid_dim, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(self.mid_dim),
            torch.nn.ReLU()
        ]
        if self.drop_rate > 0:
            img_emb_modules += [torch.nn.Dropout2d(self.drop_rate)]
        self.img_embedder = torch.nn.Sequential(*img_emb_modules)

        self.img_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = torch.nn.Linear(self.mid_dim, self.out_dim)


class TextImageLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.criterion = torch.nn.KLDivLoss(reduction='none')
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, imgs, texts):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        imgs = torch.nn.functional.softmax(imgs, dim=1)
        texts = torch.nn.functional.softmax(texts, dim=1)
        log_imgs = torch.log(imgs)
        loss = self.criterion(log_imgs, texts)
        loss = loss.sum(dim=1).mean()  
        return loss

        


class SenceDescrimnator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.loss_func = TextImageLoss()

    def text_forward(self, degrade_mat):
        return  self.text_encoder(degrade_mat)
        
    def forward(self, degrade_images, degrade_mat):
        text_code = self.text_encoder(degrade_mat)
        image_code = self.image_encoder(degrade_images)
        loss = self.loss_func(image_code, text_code)
        return loss
        

if __name__ == '__main__':
    sd = SenceDescrimnator()
    train_dataset = MultiDegradeDataset(
        '/data/lyh/train2017_degrade',
        '/data/lyh/train2017'
    )
    text = torch.tensor([[1,10,100,1,10],[1,10,100,1,10]])
    dataloader = DataLoader(train_dataset, batch_size=4)
    for  clean_images, degrade_images, degrade_mat in dataloader:
        sd(degrade_images, degrade_mat)
    