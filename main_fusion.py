import os, time
from datetime import datetime, timedelta

from train_fusion import Training
from utils.pytorchtools import EarlyStopping
from losses.focal_lovasz import FocalLosswithLovaszRegularizer
from dataloader.semantic_kitti_360 import SemanticKITTI
from models.segmentor import EncoderDecoder as Segmentor
from models.generator import EncoderDecoder as Generator


import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

class Main():
    def __init__(self):
        super(Main, self).__init__()

        self.mode = 'earlystop'
        self.name = 'fusion'
        self.train_mode = True
        self.generator_path = 'weights/generator/1215_1038exponential095_weightdecay0_SSIM1_LPIPS_Focal_stridedE_valsplit/last_weights.pth.tar'
        self.segmentor_path = 'weights/segmentor/kitti_batch6_epoch2000_FocalLosswithLovaszRegularizer_Adam/adapter_removevit_dim256_light1_1027_1039/earlystop_89_5084.pt'

        self.set_device()
        self.set_config()
        self.set_model()
        self.set_params()
        self.set_data()
        self.set_save()

    def set_config(self):
        self.epochs = 500
        if self.train_mode:
            self.batch_size = len(self.num_gpu) * 10
        else:
            self.batch_size = len(self.num_gpu) * 2
        self.nclasses = 20
        self.img_size = (384, 1248)
        self.crop_size = 128
        self.input_shape = (self.img_size[0] - self.crop_size, self.img_size[1])

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

    def set_weights(self, model, path, unfreeze=True, strict=True):
        model_info = torch.load(path)
        model.load_state_dict(model_info['model_state_dict'], strict=strict)

        for param in model.parameters():
            param.requires_grad = unfreeze

        return model

    def set_model(self):
        generator = Generator(dim=48)
        generator = DataParallel(generator.to(self.device), device_ids=self.num_gpu)
        self.generator = self.set_weights(generator, self.generator_path, unfreeze=False)

        segmentor = Segmentor(self.nclasses, self.img_size, embed_dim=256)
        segmentor = DataParallel(segmentor.to(self.device), device_ids=self.num_gpu)
        self.segmentor = self.set_weights(segmentor, self.segmentor_path, unfreeze=True, strict=False)

    def set_params(self):
        self.g_optim = Adam(self.generator.to(self.device).parameters(), lr=0.001)
        self.s_optim = Adam(self.segmentor.to(self.device).parameters(), lr=0.001)
        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optim, gamma=0.95)
        self.s_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.s_optim, gamma=0.95)

        self.criterion = FocalLosswithLovaszRegularizer(ignore_index=0)
        
    def set_data(self):
        if self.train_mode:
            self.data_path = 'data/semantic_kitti/kitti/dataset/sequences'
        else:
            self.data_path = 'data/semantic_kitti/kitti/dataset/f_sequences'

        dataset = SemanticKITTI(self.data_path, self.img_size, self.nclasses,
                                mode='train', front=360, split=False, crop_size=self.crop_size)
        dataset_size = len(dataset)
        train_size = int(dataset_size*0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def set_save(self):
        now = time.strftime('%m%d_%H%M') 
        if self.train_mode:
            self.save_path = f"weights/fusion/{str(now)}_{self.name}"
        else:
            self.save_path = f"weights/test_fusion/{str(now)}_{self.name}"

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
            epoch_log.write('\n\nepoch\ttrain loss\tval loss\ttrain miou\tval miou\ttime')
        self.metrics = {'t_loss':[], 't_miou':[], 'v_loss':[],  'v_miou':[]}    
        self.earlystop = EarlyStopping(patience=5, verbose=True, path=os.path.join(self.save_path, 'earlystop.pt'))


    def main(self):
        t_s = datetime.now()
        print(f'\ntrain start time : {t_s}')

        t = Training(self.generator, self.segmentor, self.epochs, self.train_loader, self.val_loader, self.device, 
                     self.g_optim, self.s_optim, None, self.criterion, self.g_scheduler, self.s_scheduler, 
                     self.save_path, self.earlystop, self.metrics, self.writer_train, self.writer_val, train_mode=self.train_mode)
        t.train()
        log = f'\n\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}'
        
        with open(f"{self.save_path}/result.csv", "a") as epoch_log:
            epoch_log.write(log)

if __name__ == "__main__":
    Main().main()     
         
