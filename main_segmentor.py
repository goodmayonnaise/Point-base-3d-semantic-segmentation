

import os 
import time 
from datetime import datetime, timedelta

from models.segmentor import EncoderDecoder as SalsaNextAdapter
from losses.focal_lovasz import FocalLosswithLovaszRegularizer
from dataloader.semantic_kitti_segmentor import SemanticKITTI
from utils.pytorchtools import EarlyStopping
from utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
from train_segmentor import Training 


import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
# from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

def main():

    # args --------------------------------------------------------------------------------------
    name = "segmentor"
    ckpt_path = None

    # # gpu setting -----------------------------------------------------------------------------
    torch.cuda.manual_seed_all(777)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 4
    timeout=timedelta(seconds=864000)
    dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    # setting model params --------------------------------------------------------------------
    epochs = 2000
    batch_size = len(num_gpu) * 3
    nclasses = 20 
    img_shape = (384, 1248)
    input_shape = (256, 1248)
    crop_size = 128

    # setting model ---------------------------------------------------------------------------
    model = SalsaNextAdapter(nclasses, input_shape, embed_dim=256)
    model = DataParallel(model.to(device), device_ids=num_gpu)
    if ckpt_path is not None:
        optimizer = Adam(model.to(device).parameters())
        model_info = torch.load(ckpt_path)
        model.load_state_dict(model_info['model_state_dict'], strict=True)
        for param in model.parameters():
            param.requires_grad = True
        optimizer.load_state_dict(model_info['optimizer_state_dict'])
        epoch = model_info['epoch']

    # optimizer = Adam(model.to(device).parameters(), weight_decay=0.005)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    optimizer = Adam(model.to(device).parameters(), lr=0, weight_decay=0.001)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=0.001,  T_up=10, gamma=0.5)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=75, T_mult=1, eta_max=0.001, T_up=50, gamma=0.5)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.001, T_up=75, gamma=0.5)

    criterion = FocalLosswithLovaszRegularizer(ignore_index=0)

    # setting data ----------------------------------------------------------------------------
    path = 'data/semantic_kitti/kitti/dataset/sequences'
    # train_dataset = SemanticKITTI(path, img_shape, nclasses, mode='train', front=True, split=False, crop_size=crop_size)
    # train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    # val_dataset = SemanticKITTI(path, img_shape, nclasses, mode='valid', front=True, split=False, crop_size=crop_size)
    # val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size)
    # from train_adapter_val08_salsanext_rgb import Training 
    dataset = SemanticKITTI(path, img_shape, nclasses, mode='train', front=True, split=False, crop_size=crop_size)
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size)


    # create dir for weight --------------------------------------------------------------------------------
    if ckpt_path is not None:
        model_path = ('/').join(ckpt_path.split('/')[:-1])
    else:
        configs = "{}_batch{}_epoch{}_{}_{}".format(path.split('/')[4], batch_size, epochs, str(criterion).split('(')[0], str(optimizer).split( )[0])
        print("Configs:", configs)
        now = time.strftime('%m%d_%H%M') 
        model_path = os.path.join("weights", configs, name+str(now))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    earlystop = EarlyStopping(patience=5, verbose=True, path=os.path.join(model_path, 'earlystop.pt'))

    # write log --------------------------------------------------------------------------------
    metrics = {'t_loss':[], 'v_loss':[], 't_miou':[], 'v_miou':[]}

    if ckpt_path is None:
        if not os.path.exists(os.path.join(model_path, 'samples')):
            os.makedirs(os.path.join(model_path, 'samples'))
            os.makedirs(os.path.join(model_path, 'samples', 'out1'))
            os.makedirs(os.path.join(model_path, 'samples', 'out2'))
            os.makedirs(os.path.join(model_path, 'samples', 'out2', 'train'))
            os.makedirs(os.path.join(model_path, 'samples', 'out2', 'val'))
            
        if not os.path.exists(os.path.join(model_path, 'train')):
            os.makedirs(os.path.join(model_path, 'train'))
        if not os.path.exists(os.path.join(model_path, 'val')):
            os.makedirs(os.path.join(model_path, 'val'))
 
    writer_train = SummaryWriter(log_dir=os.path.join(model_path, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(model_path, 'val'))

    with open(f'{model_path}/result.csv', 'a') as epoch_log:
        epoch_log.write('\nepoch\ttrain loss\tval loss\ttrain mIoU\tval mIoU\ttime')

    t_s = datetime.now()
    print(f'\ntrain start time : {t_s}')

    if ckpt_path is not None:
        t = Training(model, epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_path, 
                    earlystop, device, metrics, writer_train, writer_val, epoch)
    else:
        t = Training(model, epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_path, 
                    earlystop, device, metrics, writer_train, writer_val)
    t.train()

    print(f'\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}')


if __name__ == "__main__":
    main()
