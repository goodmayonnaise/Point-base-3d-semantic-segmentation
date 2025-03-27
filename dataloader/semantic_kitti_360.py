
import os, yaml, cv2, random
import numpy as np 
from glob import glob 
from einops import rearrange

from data_loader.laserscan import LaserScan, SemLaserScan
import torch
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, data_path, shape, nclasses, mode, front, split=False, crop_size=None, thetas=None, **kwargs):
        self.mode = mode
        self.front = front
        if self.front == 360:
            self.thetas = thetas
        CFG = self.load_config()
        self.swap_dict = CFG['learning_map']
        self.color_dict = CFG['color_map']
        self.learning_map = CFG['learning_map']
        self.learning_map_inv = CFG['learning_map_inv']
        # self.color_dict = {self.learning_map[key]:self.color_dict[self.learning_map_inv[self.learning_map[key]]] for key, value in self.color_dict.items()}
        sequences = CFG['split'][mode]
        self.sequences = [str(i).zfill(2) for i in sequences]
        self.path = self.data_path_load(self.sequences, data_path)

        self.scan = LaserScan(project=True, H=shape[0], W=shape[1], fov_up=3.0, fov_down=-25.0, front=self.front, crop_size=crop_size)
        self.scan = SemLaserScan(sem_color_dict=CFG['color_map'], learning_map=self.learning_map, project=True, H=shape[0], W=shape[1], fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)

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
    
    def brightness(self, img, val):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = int(random.uniform(-val, val))
        if brightness > 0:
            img = img + brightness
        else:
            img = img - brightness
        img += val
        img = np.clip(img, 10, 255)
        return img
    
    def contrast(self, img, min_val, max_val):
        #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = int(random.uniform(min_val, max_val)) # Contrast control
        adjusted = cv2.convertScaleAbs(img, alpha=alpha)
        return adjusted
    
    def saturate_contrast(self, p, num):
        pic = p.copy()
        pic = pic.astype('int64')
        pic = np.clip(pic*num, 0, 255)
        pic = pic.astype('uint8')
        return pic

    def convert_color(self, arr):
        result = np.zeros((*arr.shape, 3))
        for c in range(20):
            j = np.where(arr==c) # coord x, y
            try:
                xs, ys = j[0], j[1]
            except:
                xs = j[0]
            rgb = self.color_dict[c] # rgb 

            if len(xs) == 0:
                continue
            for x, y in zip(xs, ys):
                result[x, y] = rgb

        return result

    def __getitem__(self, idx):

        x_path, y_path, img_path = self.pcd_paths[idx], self.label_paths[idx], self.img_paths[idx]

        calib = os.path.join(('/').join(x_path.split('/')[:-2]),'calib.txt')
        img_read = cv2.imread(img_path)
        if self.front == 360:
            # for img
            img = cv2.resize(img_read, (self.input_shape[1], self.input_shape[0]))
            img = img[self.crop_size:,]
            img_half = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            img = rearrange(img, 'h w c -> c h w')
            img = torch.FloatTensor(img)
            img_half = rearrange(img_half, 'h w c -> c h w')
            img_half = torch.FloatTensor(img_half)

            rdms = torch.zeros([len(self.thetas), 3, self.input_shape[0]-self.crop_size, self.input_shape[1]])
            labels = torch.zeros([len(self.thetas), self.input_shape[0]-self.crop_size, self.input_shape[1]])
            labels_c = torch.zeros([len(self.thetas), 3, self.input_shape[0]-self.crop_size, self.input_shape[1]])
            pxs = np.zeros([len(self.thetas), 100000])
            pys = np.zeros([len(self.thetas), 100000])
            npoints = np.zeros([len(self.thetas)])
            idxs = np.zeros([len(self.thetas), 1000000])
            for idx, theta in enumerate(self.thetas):
                x = self.scan.open_scan(x_path, calib, img_read, theta=np.radians(theta))
                y = self.scan.open_label(y_path)

                rem, depth, mask = x['remission'], x['range'], x['mask']

                rdm = np.stack((rem, depth, mask), axis=0) 
                rdm = torch.FloatTensor(rdm)
                label = self.replace_with_dict(y['label'])
                label = torch.FloatTensor(label).type(torch.int64)
                label_c = torch.FloatTensor(y['label_c'])
                label_c = rearrange(label_c, 'h w c -> c h w')
                px, py = x['px'], x['py']
                npoint = px.shape[0]
                pxs[idx,:npoint], pys[idx,:npoint], npoints[idx] = px, py, npoint
                rdms[idx], labels[idx], labels_c[idx] = rdm, label, label_c
                ii = x['keep_idx']
                idxs[idx,:ii.shape[0]] = ii

            return {'rdm': rdms, 'label': labels, 'img':img, 'thetas':self.thetas, 'label_c':labels_c}
        
        elif self.front is True:
            x = self.scan.open_scan(x_path, calib, img_read)
            y = self.scan.open_label(y_path)
            img = cv2.resize(img_read, (self.input_shape[1], self.input_shape[0]))
            img = img[self.crop_size:, ]
            # img_half = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            rem, depth, mask = x['remission'], x['range'], x['mask']
            rdm = np.stack((rem, depth, mask), 0)
            rdm = torch.FloatTensor(rdm)
            depth = torch.FloatTensor(depth)

            label = self.replace_with_dict(y['label'])
            img = rearrange(img, 'h w c -> c h w')
            img = torch.FloatTensor(img)
            # img_half = rearrange(img_half, 'h w c -> c h w')
            # img_half = torch.FloatTensor(img_half)
            label_c = torch.FloatTensor(y['label_c'])
            label_c = rearrange(label_c, 'h w c -> c h w')
            unproj_range = x['unproj_range']
            px, py = x['px'], x['py']
            label_3d = torch.FloatTensor(self.replace_with_dict(y['label_3d']))

            # return {'rdm':rdm, 'label':label, 'img':img, 'img_half':img_half, 'label_c':label_c, 'label_3d':label_3d,
            return {'rdm':rdm, 'label':label, 'img':img, 'label_c':label_c, 'label_3d':label_3d,
                    'unproj_range':unproj_range, 'px':px, 'py':py, 'proj_range':depth}

    def data_path_load(self, sequences, data_path): 
        if self.mode != 'test':
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
        elif self.mode == 'test':
            pcd_names = ['/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/velodyne/000000.bin']
            label_names = ['/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/labels/000000.label']
            img_names = ['/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/image_2/000000.png']

        return pcd_names, label_names, img_names

if __name__ == "__main__":
    path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences'
    rgb_shape = (384, 1248)
    crop_size = 128
    dataset = SemanticKITTI(path, rgb_shape, 20, mode='valid', 
                            front=True, split=False, crop_size=crop_size, thetas=[0, 81, 161, 241, 321])
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for iter, batch in enumerate(loader):
        batch 