
import os, yaml, cv2
import numpy as np 
from glob import glob 
from einops import rearrange

from dataloader.laserscan_segmentor import LaserScan, SemLaserScan
import torch
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, data_path, shape, nclasses, mode, front, split=False, crop_size=None, **kwargs):
        CFG = self.load_config()
        self.swap_dict = CFG['learning_map']
        self.color_dict = CFG['color_map']
        self.learning_map = CFG['learning_map']
        self.learning_map_inv = CFG['learning_map_inv']
        sequences = CFG['split'][mode]
        self.sequences = [str(i).zfill(2) for i in sequences]
        self.path = self.data_path_load(self.sequences, data_path)
        self.scan = LaserScan(project=True, H=shape[0], W=shape[1], fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
        self.scan = SemLaserScan(sem_color_dict=CFG['color_map'], project=True, H=shape[0], W=shape[1], fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)

        self.mode = mode
        self.nclasses = nclasses
        self.input_shape = shape   
        self.split = split  
        self.crop_size = crop_size
        
        self.pcd_paths = self.path[0]
        self.label_paths = self.path[1]
        self.img_paths = self.path[2]

    def load_config(self):
        cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti2.yaml'
        try:
            print("Opening config file %s" % cfg_path)
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG

    def replace_with_dict(self, ar):
        # Extract out keys and values
        k = np.array(list(self.swap_dict.keys()))
        v = np.array(list(self.swap_dict.values()))

        # Get argsort indices
        sidx = k.argsort()
        
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]     

    def __len__(self):
        return len(self.pcd_paths)
    
    def per_class(self, t):

        per_class = torch.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]]).cuda()

        for i in range(self.nclasses):
            per_class[:,i] = torch.where(t==i, 1, 0)
        
        return per_class
    
    def __getitem__(self, idx):
        # x_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/velodyne/000001.bin'
        # y_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/labels/000001.label'
        # img_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/00/image_2/000001.png'
        x_path, y_path, img_path = self.pcd_paths[idx], self.label_paths[idx], self.img_paths[idx]
        calib = os.path.join(('/').join(x_path.split('/')[:-2]),'calib.txt')
        img_read = cv2.imread(img_path)
        x = self.scan.open_scan(x_path, calib, img_read)
        y = self.scan.open_label(y_path)

        # for img
        img = cv2.resize(img_read, (self.input_shape[1], self.input_shape[0]))
        img = img[self.crop_size:,]
        img_half = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        rem, depth, mask = x['remission'], x['range'], x['mask']
        # d = np.stack([depth, depth, depth], axis=-1)
        # cv2.imwrite('imgdepth.png', np.where(d==-1, img, d*100))
        rdm = np.stack((rem, depth, mask), axis=0) 
        if self.mode == 'test':
            cv2.imwrite('rdm.png', rearrange(rdm, 'c h w -> h w c'))
        rdm = torch.FloatTensor(rdm)
        label_3d = self.replace_with_dict(y['label'])
        if self.mode == 'test':        
            cv2.imwrite('label.png', label_3d*100)
        label_3d = torch.FloatTensor(label_3d).type(torch.int64)

        img = rearrange(img, 'h w c -> c h w')
        img = torch.FloatTensor(img)

        img_half = rearrange(img_half, 'h w c -> c h w')
        img_half = torch.FloatTensor(img_half)
        
        if self.split != False:
            rdm = rdm.unsqueeze(0)
            label_3d = label_3d.unsqueeze(0)
        if self.split==2:
            front_rdm = rdm[:,:,:,512:512*3]
            # cv2.imwrite('front_rdm2.png', front_rdm[0].transpose(2,0).transpose(1,0).cpu().detach().numpy())
            back_rdm = torch.cat([rdm[:,:,:,512*3:], rdm[:,:,:,:512]], axis=-1)
            # cv2.imwrite('back_rdm2.png', back_rdm[0].transpose(2,0).transpose(1,0).cpu().detach().numpy())
            
            front_label = label_3d[:,:,512:512*3]
            # cv2.imwrite('front_label2.png', front_label[0].cpu().detach().numpy()*100)
            back_label = torch.cat([label_3d[:,:,512*3:], label_3d[:,:,:512]], axis=-1)
            # cv2.imwrite('back_label2.png', back_label[0].cpu().detach().numpy()*100)
            
            rdms = torch.cat([front_rdm, back_rdm], axis=0)
            labels = torch.cat([front_label, back_label], axis=0)

            return {'rdm':rdms, 
                    '3d_label':labels, 
                    'img':img}
        elif self.split==3:
            rdms, labels = [], []
            for i in range(0, self.input_shape[1], 512):
                if i+1024 > self.input_shape[1]:
                    break
                # cv2.imwrite(f'samples/split/rdm/rdm_2to3split{i}.png', rearrange(rdm[0,:,:,i:i+1024].numpy(), 'c h w -> h w c'))
                # cv2.imwrite(f'samples/split/label/label_2to3split{i}.png', label_3d[0,:,i:i+1024].numpy()*100)
                rdms.append(rdm[:,:,:,i:i+1024])
                labels.append(label_3d[:,:,i:i+1024])
            return {'rdm':torch.cat([rdms[i] for i in range(len(rdms))], axis=0), 
                    '3d_label':torch.cat([labels[i] for i in range(len(labels))], axis=0), 
                    'img':img}
            
        return {'rdm':rdm, '3d_label':label_3d, 'label_c':y['label_c'], 'img':img, 'img_half':img_half}

        # rem = x['remission']
        # rem = torch.FloatTensor(rem)
        # rem = torch.unsqueeze(rem, 0)

        # pad_rem = torch.FloatTensor(x['pad_remission'])
        # pad_rem = torch.unsqueeze(pad_rem,0)
        # img_half = cv2.resize(img_read, (int(self.input_shape[1]/2), int(self.input_shape[0]/2)))
        # img_half = rearrange(img_half, 'h w c -> c h w')
        # img_half = torch.FloatTensor(img_half)

        # y = torch.FloatTensor(y['label'])
        # h, w = y.shape 
        # y_class = torch.zeros(self.nclasses, h, w)
        # for c in range(self.nclasses):
        #     y_class[c] = (y==c).type(torch.int32).clone().detach()
        # return {'rdm':rdm, '3d_label_c':self.per_class(label_3d)}
        # return {'img':img, 'img_half':img_half, 'rem':rem, 'rem255':rem*255}
        # return {'img': img, 'img_half':img_half, 'rem':rem}
        # return {'img': img, 'label':y_class, 'rgb_label':y, 'rem':rem, 'pad_rem':pad_rem}

    def data_path_load(self, sequences, data_path): 
        img_paths = [os.path.join(data_path, sequence_num, "image_2") for sequence_num in sequences]
        pcd_paths = [os.path.join(data_path, sequence_num, "velodyne") for sequence_num in sequences]
        label_paths = [os.path.join(data_path, sequence_num, "labels") for sequence_num in sequences]
        
        pcd_names, label_names, img_names = [], [], []

        for pcd_path, label_path, img_path in zip(pcd_paths, label_paths, img_paths):    
            pcd_names = pcd_names + glob(str(os.path.join(os.path.expanduser(pcd_path),"*.bin")))
            label_names = label_names + glob(str(os.path.join(os.path.expanduser(label_path),"*.label")))
            img_names = img_names + glob(str(os.path.join(os.path.expanduser(img_path),"*.png")))

        pcd_names.sort()
        label_names.sort()
        img_names.sort()

        return pcd_names, label_names, img_names


if __name__ == "__main__":
    def load_config():
        cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
        try:
            print("Opening config file %s" % "config/semantic-kitti.yaml")
            import yaml
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG
    from laserscan_0920 import LaserScan, SemLaserScan
    from torch.utils.data import DataLoader
    import cv2, torch 
    data_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences'
    shape = (384, 1248) 
    mode = "train"
    nclasses = 20 
    batch_size = 1
    dataset = SemanticKITTI(data_path, shape, nclasses, mode, front=True, split=False, crop_size=128)

    loader = DataLoader(dataset, batch_size, num_workers=1, shuffle=True)
    

    from einops import rearrange
    for iter, batch in enumerate(loader):
        inputs = batch['rdm']
        labels = batch['3d_label']
        imgs = batch['img']
        
        # inputs = rearrange(batch['rdm'], 'b1 b2 c h w -> (b1 b2) c h w')
        # labels = rearrange(batch['3d_label'], 'b1 b2 h w -> (b1 b2) h w')
        # imgs = batch['img']

        # out_segment = torch.rand([batch_size, 20, 256, 1024])

        # get_iou = IOUEval(nclasses, ignore=0)
        # get_iou.addBatch(torch.argmax(out_segment, 1), labels)
        # iou_mean, per_iou = get_iou.getIoU()        
        
        print()
        
    