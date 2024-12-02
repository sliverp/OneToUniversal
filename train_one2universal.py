import os, time, torch, argparse
from torch.utils.data import DataLoader
from utils.utils import print_tensor_shape, load_restore_ckpt_with_optim, load_embedder_ckpt, adjust_learning_rate, data_process, tensor_metric, load_excel, save_checkpoint


from utils.utils import compare_ssim, compare_psnr, compare_mse

import time
import torch
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sence_descrimnator.dataset import MultiDegradeDataset
from sence_descrimnator.model import SenceDescrimnator
from model.OneRestore import OneRestore
import argparse 
from pathlib import Path
from model.loss import Total_loss as LossModel

train_config = {
    "train":{
        "epoch": 200,
        "batch_size": 2,
        "lr": 5e-4,
        "resume": "",
        "embedder_path": "/home/liyh/OneToUniversal/sence_descrimnator/exp/desc/Exp2_nomlp_res101/best_0.03769.pt",
        "save_path":"./exp/restore/{}/",
        "save_epoch": 1,
        "train_log": "./exp/restore/{}/",
        "ckpt_keep": 3,
        "worker": 6,
    },
     "test":{
        "batch_size": 2,
        "save_epoch": 1,
        "test_interval": 1,
        "worker": 0
    }
}


def load_restore_ckpt_with_optim(save_path, ckpt_name, embedder_path, local_rank=None, lr=None):
    if not embedder_path:
        print("no embedder!!")
        embedder = None
    else:
        embedder = SenceDescrimnator().to("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            embedder_info = torch.load(embedder_path)
        else:
            embedder_info = torch.load(embedder_path, map_location=torch.device('cpu'))
        embedder.load_state_dict(embedder_info['state_dict'], strict=False)
        
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    
    if ckpt_name:
        if torch.cuda.is_available():
            model_info = torch.load(os.path.join(save_path, ckpt_name))
        else:
            model_info = torch.load(os.path.join(save_path, ckpt_name), map_location=torch.device('cpu'))

        print('==> loading existing model:', ckpt_name)
        model = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
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
        model = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if local_rank != None else torch.nn.DataParallel(model)
        cur_epoch = 0


    total = sum([param.nelement() for param in model.parameters()])
    print("Number of SenceDescrimnator parameter: %.2fM" % (total/1e6))

    return model, embedder, optimizer, cur_epoch

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
        }, file_path)
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



def metric_caculater():
    psnrs = []
    ssims = []
    mses = []
    def caulate(gt, result, type):
        nonlocal psnrs, ssims, mses
        gt = gt.detach()
        result = result.detach()
        if type == "PSNR":
            psnrs.append(compare_psnr(gt, result))
            return psnrs
        elif type == "SSIM":
            ssims.append(compare_ssim(gt, result))
            return ssims
        elif type == "MSE":
            mses.append(compare_mse(gt, result))
            return mses
    return caulate

def carry_data_to_cuda(*args):
    for data in args:
        data.cuda()

def start_train(comment, is_debug):
    dataset = MultiDegradeDataset('/home/liyh/train2017_degrade', '/home/liyh/train2017', contrastive=True)
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
    model, embedder, optimizer, cur_epoch = load_restore_ckpt_with_optim(
        save_path=train_config['train']['save_path'].format(comment),
        ckpt_name=train_config['train']['resume'], 
        embedder_path=train_config['train']['embedder_path'],
        lr=train_config['train']['lr']
    )
    ckpt_save = save_last_and_best_ckpt()
    caculate_metric = metric_caculater()
    loss_func = LossModel()
    for epoch in range(cur_epoch, train_config['train']['epoch']):
        model.train()
        bar = tqdm(train_loader)
        total_loss = []
        for clean_images, degrade_images, negtive_images, degrade_mat in bar:
            # print_tensor_shape("clean_images", clean_images)
            # print_tensor_shape("degrade_images", degrade_images)
            # print_tensor_shape("negtive_images", negtive_images)
            degrade_mat = degrade_mat.cuda()
            optimizer.zero_grad()
            if embedder is not None:
                text_embedding = embedder.text_forward(degrade_mat)
            else:
                text_embedding = torch.ones(size=(degrade_images.shape[0], 324))
            degrade_images, clean_images = degrade_images.cuda(), clean_images.cuda()
            negtive_images = negtive_images.cuda()
            
            result = model(degrade_images, text_embedding)
            loss = loss_func(degrade_images, clean_images, negtive_images, result)
            loss.backward()
            optimizer.step()

            psnrs = caculate_metric(clean_images, result, "PSNR")
            ssims = caculate_metric(clean_images, result, "SSIM")
            mses = caculate_metric(clean_images, result, "MSE")
            total_loss.append(loss.item())

            bar.set_description(
                f"[epoch {epoch}]|loss:{loss.item():.5f}|MSE:{mses[-1]:.5f}|PSNR:{psnrs[-1]:.4f}|SSIM:{ssims[-1]:.4f}"
            )
            if is_debug: break

        print(f"eopch[{epoch}] train avg loss: {sum(total_loss)/len(total_loss)}")
        if epoch % train_config['train']['save_epoch'] == 0:
            ckpt_save(comment, epoch, model, optimizer, loss.item())

        # if epoch % train_config["test"]['test_interval'] == 0:
        #     model.eval()
        #     bar = tqdm(test_loader)
        #     total_loss = []
        #     for clean_images, degrade_images, degrade_mat in bar:
        #         degrade_images, degrade_mat = degrade_images.cuda(), degrade_mat.cuda()
        #         loss = model(degrade_images, degrade_mat)
        #         bar.set_description(f"loss:{loss.sum().item()}")
        #         total_loss.append(loss.sum().item())
        #         if is_debug: break
        #     print(f"eopch[{epoch}] test avg loss: {sum(total_loss)/len(total_loss)}")
        if is_debug:
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Start Training")
    parser.add_argument("--comment", type=str, default="Exp2_degrade_mat", help='train name')
    parser.add_argument("--debug", action="store_true", help='debug mode')
    argspar = parser.parse_args()
    start_train(argspar.comment, argspar.debug)
    