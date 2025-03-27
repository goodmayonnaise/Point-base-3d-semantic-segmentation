
import os, yaml
import numpy as np 

from tqdm import tqdm 
from datetime import timedelta

from data_loader.semantic_kitti_360_validation import SemanticKITTI

from torch.utils.data import DataLoader

def main():
    out_bin_dir = 'weights/generator/0321_0957_256x1248_NOTrdmscale_transferlearning_base_after_pj_transfer_learning2_val08_notsync_deform_padlabel_aug_cntx/testseq/val3_best_lpips_epoch76/sequences/08/predictions'
    CFG = yaml.safe_load(open('data_loader/semantic-kitti.yaml', 'r'))
    color_dict = CFG['color_map']
    learning_map = CFG['learning_map']
    learning_map_inv = CFG['learning_map_inv']
    color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

    dataset = SemanticKITTI(data_path='/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences',
                            shape=(384,1248*5),
                            nclasses=20, 
                            mode='test',
                            front=False,
                            crop_size=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)



    for idx, batch in enumerate(loader):
        
        pcd = batch['pcd']
        x_path = batch['x_path']
        px = batch['px']

        out_bin_path = os.path.join(out_bin_dir, x_path[0].split('/')[-3], x_path[0].split('/')[-1])
        out_bin_load = np.fromfile(out_bin_path)

        with open(('/').join(out_bin_dir.split('/')[:-2]+['assert_list.txt']), 'w+') as f:
            f.write(f"out bin path, x_path")

            print(x_path, '\n', out_bin_load.shape[0], px.shape[-1], pcd.shape[1])
            if (out_bin_load.shape[0] != px.shape[-1]) or (out_bin_load.shape[0]!=pcd.shape[1]):
                f.write(f"\n{out_bin_path}\t{x_path}")
                

if __name__=='__main__':


    main()