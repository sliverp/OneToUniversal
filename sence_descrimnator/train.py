import time
import torch
import os
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from tqdm import tqdm
from PIL import Image
from dataset import MultiDegradeDataset
from model import SenceDescrimnator
import argparse 
from pathlib import Path


train_config = {
    "train":{
        "epoch": 200,
        "batch_size": 1,
        "lr": 1e-5,
        "resume": "",
        "save_path":"./exp/desc/{}/",
        "save_epoch": 1,
        "train_log": "./exp/desc/{}/",
        "ckpt_keep": 3,
        "worker": 16,
    },
     "test":{
        "batch_size": 1,
        "save_path":"./exp/desc/{}/",
        "save_epoch": 1,
        "train_log": "./exp/desc/{}/",
        "test_interval": 1,
        "worker": 0
    }
}


def load_restore_ckpt_with_optim(save_path, ckpt_name, local_rank=None, freeze_model=False, lr=None):
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    if ckpt_name:
        if torch.cuda.is_available():
            model_info = torch.load(os.path.join(save_path, ckpt_name))
        else:
            model_info = torch.load(os.path.join(save_path, ckpt_name), map_location=torch.device('cpu'))

        print('==> loading existing model:', ckpt_name)
        model = SenceDescrimnator().to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) if lr != None else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if local_rank != None else model

        if local_rank != None:
            model.load_state_dict(model_info['state_dict'])
        else:
            weights_dict = {}
            for k, v in model_info['state_dict'].items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
            model.load_state_dict(weights_dict)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']
    else:
        print('==> Initialize model.')
        model = SenceDescrimnator().to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if local_rank != None else torch.nn.DataParallel(model)
        cur_epoch = 0


    total = sum([param.nelement() for param in model.parameters()])
    print("Number of SenceDescrimnator parameter: %.2fM" % (total/1e6))

    return model, optimizer, cur_epoch

def save_last_and_best_ckpt():
    saved_ckpts = []
    bast_loss = 9999999
    last_bast_path = ''
    def save(comment, epoch, model, optimizer, loss):
        nonlocal last_bast_path, bast_loss, saved_ckpts
        file_name = f"{epoch}_{loss:.5f}.pt"
        file_path  = train_config['train']['save_path'].format(comment) + file_name
        if len(saved_ckpts) > train_config['train']['ckpt_keep']:
            need_delete = saved_ckpts.pop(0)
            os.remove(need_delete)
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }, file_path
        )
        saved_ckpts.append(file_path)
        if not last_bast_path or loss < bast_loss:
            bast_loss = loss
            if last_bast_path != '':
                os.remove(last_bast_path)
            last_bast_path = train_config['train']['save_path'].format(comment) + f"best_{loss:.5f}.pt"
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                }, last_bast_path
            )
    return save




def start_train(comment, is_debug):
    dataset = MultiDegradeDataset('/home/liyh/train2017_degrade', '/home/liyh/train2017')
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_set, 
        batch_size=train_config["train"]["batch_size"], shuffle=True, 
        num_workers=train_config["train"]["worker"]
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=train_config["test"]["batch_size"], shuffle=False,
        num_workers=train_config["train"]["worker"]
    )

    print(train_config['train']['save_path'].format(comment))
    model, optimizer, cur_epoch = load_restore_ckpt_with_optim(
        save_path=train_config['train']['save_path'].format(comment),
        ckpt_name=train_config['train']['resume'], 
        lr=train_config['train']['lr']
    )
    ckpt_save = save_last_and_best_ckpt()
    for epoch in range(cur_epoch, train_config['train']['epoch']):
        model.train()
        bar = tqdm(train_loader)
        total_loss = []
        for clean_images, degrade_images, degrade_mat in bar:
            # torchvision.utils.save_image(clean_images,'./clean.jpg')
            # torchvision.utils.save_image(degrade_images,'./degrade.jpg')
            # print(degrade_mat)
            optimizer.zero_grad()
            degrade_images, degrade_mat = degrade_images.cuda(), degrade_mat.cuda()
            loss = model(degrade_images, degrade_mat)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            bar.set_description(f"loss:{loss.item()}")
            if is_debug: break
        print(f"eopch[{epoch}] train avg loss: {sum(total_loss)/len(total_loss)}")
        if epoch % train_config['train']['save_epoch'] == 0:
            ckpt_save(comment, epoch, model, optimizer, loss.item())

        if epoch % train_config["test"]['test_interval'] == 0:
            model.eval()
            bar = tqdm(test_loader)
            total_loss = []
            for clean_images, degrade_images, degrade_mat in bar:
                degrade_images, degrade_mat = degrade_images.cuda(), degrade_mat.cuda()
                loss = model(degrade_images, degrade_mat)
                bar.set_description(f"loss:{loss.sum().item()}")
                total_loss.append(loss.sum().item())
                if is_debug: break
            print(f"eopch[{epoch}] test avg loss: {sum(total_loss)/len(total_loss)}")
        if is_debug:
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Start Training")
    parser.add_argument("--comment", type=str, default="derect_classify_nofrozen_1024-512-256-5", help='train name')
    parser.add_argument("--debug", action="store_true", help='debug mode')
    argspar = parser.parse_args()
    start_train(argspar.comment, argspar.debug)