
import os
from datetime import datetime, timedelta
import time 

from losses.lpips import LPIPS
from losses.focal_lovasz import FocalLosswithLovaszRegularizer
from losses.ssim import SSIM
from dataloader.semantic_kitti_360 import SemanticKITTI
from train import Training
from models.generator import EncoderDecoder as Generator 

import torch 
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class MAIN():
    def __init__(self):
        super(MAIN, self).__init__()

        self.mode = 'epoch'
        self.name = 'generator_mfdscm_rdmscale255'
        self.train_mode = True
        self.set_device()
        self.set_config()
        self.set_model()
        self.set_params()
        self.set_data()
        self.set_save()

    def set_config(self):
        self.epochs = 150
        if self.train_mode:
            self.batch_size = len(self.num_gpu) * 10
        else:
            self.batch_size = len(self.num_gpu) * 2
        self.nclasses = 20
        self.img_size = (384, 1248)
        self.crop_size = 128
        self.input_shape = (self.img_size[0] - self.crop_size, self.img_size[1])
        self.earlystop = None

    def set_device(self):
        torch.cuda.manual_seed_all(777)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        self.num_gpu = list(range(torch.cuda.device_count()))
        self.num_workers = len(gpus.split(",")) * 4
        timeout = timedelta(seconds=864000)
        dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    def set_model(self):
        model = Generator(dim=48)
        self.model = DataParallel(model.to(self.device), device_ids=self.num_gpu)

    def set_params(self):
        self.optimizer = Adam(self.model.to(self.device).parameters(), lr=0.001, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.criterion = SSIM()
        self.criterion2 = LPIPS(net='vgg').to(self.device)
        self.criterion3 = FocalLosswithLovaszRegularizer(ignore_index=0)
    
    def set_data(self):
        if self.train_mode :
            self.path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences'
        else:
            self.path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/f_sequences'

        dataset = SemanticKITTI(self.path, self.img_size, self.nclasses, 
                                mode='train', front=True, split=False, crop_size=self.crop_size)
        dataset_size = len(dataset)
        train_size = int(dataset_size*0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

       
    def set_save(self):
        now = time.strftime('%m%d_%H%M') 
        if self.train_mode:
            self.save_path = f"weights/generator/{str(now)}_{self.name}"
        else:
            self.save_path = f"weights/test_generator/{str(now)}_{self.name}"
            # self.train_mode = True

        if not os.path.exists(os.path.join(self.save_path)):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, 'samples', 'train'))
            os.makedirs(os.path.join(self.save_path, 'samples', 'val'))
            os.makedirs(os.path.join(self.save_path, 'train'))
            os.makedirs(os.path.join(self.save_path, 'val'))

        self.writer_train = SummaryWriter(log_dir=os.path.join(self.save_path, 'train'))
        self.writer_val = SummaryWriter(log_dir = os.path.join(self.save_path, 'val'))

        with open(f'{self.save_path}/result.csv', 'a') as epoch_log:
            epoch_log.write(f"{self.name+str(now)}_batch{self.batch_size}_{str(self.criterion).split('(')[0]}")
            epoch_log.write('\n\nepoch\ttrain loss\tval loss\ttrain lpips\tval lpips\ttrain psnr\tval psnr\ttrain ssim\tval ssim\ttime')
        self.metrics = {'t_loss':[], 't_psnr':[], 't_ssim':[], 't_lpips':[], 't_segment':[], 't_miou':[],
                        'v_loss':[], 'v_psnr':[], 'v_ssim':[], 'v_lpips':[], 'v_segment':[], 'v_miou':[]}    

    def main(self):
        t_s = datetime.now()
        print(f'\ntrain start time : {t_s}')

        t = Training(self.model, self.epochs, self.train_loader, self.val_loader, self.device, self.optimizer, self.criterion, self.criterion2, self.criterion3,
                     self.scheduler, self.save_path, self.earlystop, self.metrics, self.writer_train, self.writer_val, mode=self.mode, train_mode=self.train_mode)
        t.start_train()
        log = f'\n\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}'
        
        with open(f"{self.save_path}/result.csv", "a") as epoch_log:
            epoch_log.write(log)


if __name__ == "__main__":
    MAIN().main()