import warnings
import numpy as np
import torch
import os
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import pandas as pd

from model.OneRestore import OneRestore
from model.Embedder import Embedder

import torch.nn.functional as F

def load_embedder_ckpt(device, freeze_model=False, ckpt_name=None,
                                  combine_type = ['clear', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']):
    if ckpt_name != None:
        if torch.cuda.is_available():
            model_info = torch.load(ckpt_name)
        else:
            model_info = torch.load(ckpt_name, map_location=torch.device('cpu'))

        print('==> loading existing Embedder model:', ckpt_name)
        model = Embedder(combine_type)
        model.load_state_dict(model_info)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    else:
        print('==> Initialize Embedder model.')
        model = Embedder(combine_type)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    if freeze_model:
        freeze(model)

    return model

def load_restore_ckpt(device, freeze_model=False, ckpt_name=None):
    if ckpt_name != None:
        if torch.cuda.is_available():
            model_info = torch.load(ckpt_name)
        else:
            model_info = torch.load(ckpt_name, map_location=torch.device('cpu'))
        print('==> loading existing OneRestore model:', ckpt_name)
        model = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(model_info)
    else:
        print('==> Initialize OneRestore model.')
        model = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model).to("cuda" if torch.cuda.is_available() else "cpu")

    if freeze_model:
        freeze(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of OneRestore parameter: %.2fM" % (total/1e6))

    return model

def load_restore_ckpt_with_optim(device, local_rank=None, freeze_model=False, ckpt_name=None, lr=None):
    if ckpt_name != None:
        if torch.cuda.is_available():
            model_info = torch.load(ckpt_name)
        else:
            model_info = torch.load(ckpt_name, map_location=torch.device('cpu'))

        print('==> loading existing OneRestore model:', ckpt_name)
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
        print('==> Initialize OneRestore model.')
        model = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if local_rank != None else torch.nn.DataParallel(model)
        cur_epoch = 0

    if freeze_model:
        freeze(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of OneRestore parameter: %.2fM" % (total/1e6))

    return model, optimizer, cur_epoch

def load_embedder_ckpt_with_optim(device, args, combine_type = ['clear', 'low', 'haze', 'rain', 'snow',\
    'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow']):
    print('Init embedder')
    # seed
    if args.seed == -1:
        args.seed = np.random.randint(1, 10000)
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Training embedder seed:', seed)

    # embedder model
    embedder = Embedder(combine_type).to("cuda" if torch.cuda.is_available() else "cpu")

    if args.pre_weight == '':
        optimizer = torch.optim.Adam(embedder.parameters(), lr=args.lr)
        cur_epoch = 1
    else:
        try:
            embedder_info = torch.load(f'{args.check_dir}/{args.pre_weight}')
            if torch.cuda.is_available():
                embedder_info = torch.load(f'{args.check_dir}/{args.pre_weight}')
            else:
                embedder_info = torch.load(f'{args.check_dir}/{args.pre_weight}', map_location=torch.device('cpu'))
            embedder.load_state_dict(embedder_info['state_dict'])
            optimizer = torch.optim.Adam(embedder.parameters(), lr=args.lr)
            optimizer.load_state_dict(embedder_info['optimizer'])
            cur_epoch = embedder_info['epoch'] + 1
        except:
            print('Pre-trained model loading error!')
    return embedder, optimizer, cur_epoch, device

def freeze_text_embedder(m):
    """Freezes module m.
    """
    m.eval()
    for name, para in m.named_parameters():
        if name == 'embedder.weight' or name == 'mlp.0.weight' or name == 'mlp.0.bias':
            print(name)
            para.requires_grad = False
            para.grad = None

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def data_process(data, args, device):
    combine_type = args.degr_type
    b,n,c,w,h = data.size()

    pos_data = data[:,0,:,:,:]

    inp_data = torch.zeros((b,c,w,h))
    inp_class = []

    neg_data = torch.zeros((b,n-2,c,w,h))

    index = np.random.randint(1, n, (b))
    for i in range(b):
        k = 0
        for j in range(n):
            if j == 0:
                continue
            elif index[i] == j:
                inp_class.append(combine_type[index[i]])
                inp_data[i, :, :, :] = data[i, index[i], :, :,:]
            else:
                neg_data[i,k,:,:,:] = data[i, j, :, :,:]
                k=k+1
    return pos_data.to("cuda" if torch.cuda.is_available() else "cpu"), [inp_data.to("cuda" if torch.cuda.is_available() else "cpu"), inp_class], neg_data.to("cuda" if torch.cuda.is_available() else "cpu")

def print_args(argspar):
    print("\nParameter Print")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] /2
    return optimizer


def tensor_metric(img, imclean, model, data_range=1):

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True)
            # due to the skimage vision problem, you can replace above line by
            # SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, channel_axis=-1)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]

def save_checkpoint(stateF, checkpoint, epoch, psnr_t1,ssim_t1,psnr_t2,ssim_t2, filename='model.tar'):
    torch.save(stateF, checkpoint + 'OneRestore_model_%d_%.4f_%.4f_%.4f_%.4f.tar'%(epoch,psnr_t1,ssim_t1,psnr_t2,ssim_t2))

def load_excel(x):
    data1 = pd.DataFrame(x)

    writer = pd.ExcelWriter('./mertic_result.xlsx')	
    data1.to_excel(writer, 'PSNR-SSIM', float_format='%.5f')
    # writer.save()
    writer.close()

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def compare_mse(image0, image1):
    return torch.mean((image0 - image1) ** 2)

def compare_psnr(pred_image, gt):
    assert pred_image.shape == gt.shape
    diff = pred_image - gt

    rmse = (diff ** 2.).mean().sqrt()
    return 20*torch.log10(1.0/rmse)

def print_tensor_shape(name, tensor:torch.Tensor):
    print(f"{name} shape:{tensor.shape}")



def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs



def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def compare_ssim(
    X,
    Y,
    data_range=1.0,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)
    