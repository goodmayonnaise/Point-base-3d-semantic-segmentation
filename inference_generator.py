
import cv2, os, time, yaml
import numpy as np
from einops import rearrange
from datetime import timedelta

from utils.knn import KNN
from utils.metrics import IOUEval
from utils.logs import AverageMeter, ProgressMeter
from losses.focal_lovasz import FocalLosswithLovaszRegularizer
from dataloader.semantic_kitti_knn import SemanticKITTI
from models.segmentor import EncoderDecoder as Segmentor
from models.generator import EncoderDecoder as Generator

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

class Inference():
    def __init__(self):
        super(Inference, self).__init__()
        self.generator_path = 'weights/generator/1215_1038exponential095_weightdecay0_SSIM1_LPIPS_Focal_stridedE_valsplit/last_weights.pth.tar'
        self.segmentor_path = 'weights/fusion/1220_1640_base'

        self.set_cuda()
        self.thetas = [81, 241]
        self.set_loader()

        self.crop_size = 128
        self.knn = KNN()
        # self.knn = KNN(search=21)

    def set_cuda(self):
        torch.cuda.manual_seed_all(777)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        self.num_gpu = list(range(torch.cuda.device_count()))
        self.num_workers = len(gpus.split(",")) * 2
        timeout = timedelta(seconds=864000)
        dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    def set_model(self):
        generator = Generator()
        self.generator = DataParallel(generator.to(self.device), device_ids=self.num_gpu)
        g_info = torch.load(self.generator_path)
        self.generator.load_state_dict(g_info['model_state_dict'], strict=True)
        for param in self.generator.parameters():
            param.requires_grad = False

        segmentor = Segmentor(20, (256, 1248), embed_dim=256)
        self.segmentor = DataParallel(segmentor.to(self.device), device_ids=self.num_gpu)
        s_info = torch.load(self.ckpt_path)
        self.segmentor.load_state_dict(s_info['model_state_dict'], strict=True)
        for param in self.segmentor.parameters():
            param.requires_grad = False

        self.epoch = s_info['epoch']
        self.mode = self.ckpt_path.split('/')[-1].split('.')[0]
        if self.mode == 'best_miou':
            self.miou = str(s_info['best_psnr'].item())[2:6]
        elif self.mode == 'last_weights':
            self.miou = str(s_info['last_miou'].item())[2:6]
        elif self.mode == 'earlystop':
            self.miou = 'none'

        self.criterion = FocalLosswithLovaszRegularizer(ignore_index=0)

    def set_init(self):

        self.loss1_run = AverageMeter("Loss1", ":.4f")
        self.loss2_run = AverageMeter("Loss2", ":.4f")

        self.miou1_run = AverageMeter("mIoU1", ":.4f")
        self.miou2_run = AverageMeter("mIoU2", ":.4f")
        self.miou3_run = AverageMeter("3D_mIoU", ":.4f")
        self.miou4_run = AverageMeter("3D_mIoU", ":.4f")
        self.miou5_run = AverageMeter("3D_mIoU_KNN", ":.4f")
        self.miou6_run = AverageMeter("3D_mIoU_KNN", ":.4f")

        self.car1_run = AverageMeter('car', ':.4f')
        self.car2_run = AverageMeter('car', ':.4f')
        self.car3_run = AverageMeter('car', ':.4f')
        self.car4_run = AverageMeter('car', ':.4f')
        self.car5_run = AverageMeter('car', ':.4f')
        self.car6_run = AverageMeter('car', ':.4f')

        self.bicycle1_run = AverageMeter('bicycle', ':.4f')
        self.bicycle2_run = AverageMeter('bicycle', ':.4f')
        self.bicycle3_run = AverageMeter('bicycle', ':.4f')
        self.bicycle4_run = AverageMeter('bicycle', ':.4f')
        self.bicycle5_run = AverageMeter('bicycle', ':.4f')
        self.bicycle6_run = AverageMeter('bicycle', ':.4f')

        self.motorcycle1_run = AverageMeter('motorcycle', ':.4f')
        self.motorcycle2_run = AverageMeter('motorcycle', ':.4f')
        self.motorcycle3_run = AverageMeter('motorcycle', ':.4f')
        self.motorcycle4_run = AverageMeter('motorcycle', ':.4f')
        self.motorcycle5_run = AverageMeter('motorcycle', ':.4f')
        self.motorcycle6_run = AverageMeter('motorcycle', ':.4f')

        self.truck1_run = AverageMeter('trunk', ':.4f')
        self.truck2_run = AverageMeter('trunk', ':.4f')
        self.truck3_run = AverageMeter('trunk', ':.4f')
        self.truck4_run = AverageMeter('trunk', ':.4f')
        self.truck5_run = AverageMeter('trunk', ':.4f')
        self.truck6_run = AverageMeter('trunk', ':.4f')

        self.other_vehicle1_run = AverageMeter('other_vehicle', ':.4f')
        self.other_vehicle2_run = AverageMeter('other_vehicle', ':.4f')
        self.other_vehicle3_run = AverageMeter('other_vehicle', ':.4f')
        self.other_vehicle4_run = AverageMeter('other_vehicle', ':.4f')
        self.other_vehicle5_run = AverageMeter('other_vehicle', ':.4f')
        self.other_vehicle6_run = AverageMeter('other_vehicle', ':.4f')

        self.person1_run = AverageMeter('person', ':.4f')
        self.person2_run = AverageMeter('person', ':.4f')
        self.person3_run = AverageMeter('person', ':.4f')
        self.person4_run = AverageMeter('person', ':.4f')
        self.person5_run = AverageMeter('person', ':.4f')
        self.person6_run = AverageMeter('person', ':.4f')

        self.bicyclist1_run = AverageMeter('bicyclist', ':.4f')
        self.bicyclist2_run = AverageMeter('bicyclist', ':.4f')
        self.bicyclist3_run = AverageMeter('bicyclist', ':.4f')
        self.bicyclist4_run = AverageMeter('bicyclist', ':.4f')
        self.bicyclist5_run = AverageMeter('bicyclist', ':.4f')
        self.bicyclist6_run = AverageMeter('bicyclist', ':.4f')

        self.motorcyclist1_run = AverageMeter('motorcyclist', ':.4f')
        self.motorcyclist2_run = AverageMeter('motorcyclist', ':.4f')
        self.motorcyclist3_run = AverageMeter('motorcyclist', ':.4f')
        self.motorcyclist4_run = AverageMeter('motorcyclist', ':.4f')
        self.motorcyclist5_run = AverageMeter('motorcyclist', ':.4f')
        self.motorcyclist6_run = AverageMeter('motorcyclist', ':.4f')

        self.road1_run = AverageMeter('road', ':.4f')
        self.road2_run = AverageMeter('road', ':.4f')
        self.road3_run = AverageMeter('road', ':.4f')
        self.road4_run = AverageMeter('road', ':.4f')
        self.road5_run = AverageMeter('road', ':.4f')
        self.road6_run = AverageMeter('road', ':.4f')

        self.parking1_run = AverageMeter('parking', ':.4f')
        self.parking2_run = AverageMeter('parking', ':.4f')
        self.parking3_run = AverageMeter('parking', ':.4f')
        self.parking4_run = AverageMeter('parking', ':.4f')
        self.parking5_run = AverageMeter('parking', ':.4f')
        self.parking6_run = AverageMeter('parking', ':.4f')
        
        self.sidewalk1_run = AverageMeter('sidewalk', ':.4f')
        self.sidewalk2_run = AverageMeter('sidewalk', ':.4f')
        self.sidewalk3_run = AverageMeter('sidewalk', ':.4f')
        self.sidewalk4_run = AverageMeter('sidewalk', ':.4f')
        self.sidewalk5_run = AverageMeter('sidewalk', ':.4f')
        self.sidewalk6_run = AverageMeter('sidewalk', ':.4f')

        self.other_ground1_run = AverageMeter('other_ground', ':.4f')
        self.other_ground2_run = AverageMeter('other_ground', ':.4f')
        self.other_ground3_run = AverageMeter('other_ground', ':.4f')
        self.other_ground4_run = AverageMeter('other_ground', ':.4f')
        self.other_ground5_run = AverageMeter('other_ground', ':.4f')
        self.other_ground6_run = AverageMeter('other_ground', ':.4f')

        self.building1_run = AverageMeter('building', ':.4f')
        self.building2_run = AverageMeter('building', ':.4f')
        self.building3_run = AverageMeter('building', ':.4f')
        self.building4_run = AverageMeter('building', ':.4f')
        self.building5_run = AverageMeter('building', ':.4f')
        self.building6_run = AverageMeter('building', ':.4f')

        self.fence1_run = AverageMeter('fence', ':.4f')
        self.fence2_run = AverageMeter('fence', ':.4f')
        self.fence3_run = AverageMeter('fence', ':.4f')
        self.fence4_run = AverageMeter('fence', ':.4f')
        self.fence5_run = AverageMeter('fence', ':.4f')
        self.fence6_run = AverageMeter('fence', ':.4f')

        self.vegetation1_run = AverageMeter('vegetation', ':.4f')
        self.vegetation2_run = AverageMeter('vegetation', ':.4f')
        self.vegetation3_run = AverageMeter('vegetation', ':.4f')
        self.vegetation4_run = AverageMeter('vegetation', ':.4f')
        self.vegetation5_run = AverageMeter('vegetation', ':.4f')
        self.vegetation6_run = AverageMeter('vegetation', ':.4f')

        self.trunk1_run = AverageMeter('trunk', ':.4f')
        self.trunk2_run = AverageMeter('trunk', ':.4f')
        self.trunk3_run = AverageMeter('trunk', ':.4f')
        self.trunk4_run = AverageMeter('trunk', ':.4f')
        self.trunk5_run = AverageMeter('trunk', ':.4f')
        self.trunk6_run = AverageMeter('trunk', ':.4f')

        self.terrain1_run = AverageMeter('terrain', ':.4f')
        self.terrain2_run = AverageMeter('terrain', ':.4f')
        self.terrain3_run = AverageMeter('terrain', ':.4f')
        self.terrain4_run = AverageMeter('terrain', ':.4f')
        self.terrain5_run = AverageMeter('terrain', ':.4f')
        self.terrain6_run = AverageMeter('terrain', ':.4f')

        self.pole1_run = AverageMeter('pole', ':.4f')
        self.pole2_run = AverageMeter('pole', ':.4f')
        self.pole3_run = AverageMeter('pole', ':.4f')
        self.pole4_run = AverageMeter('pole', ':.4f')
        self.pole5_run = AverageMeter('pole', ':.4f')
        self.pole6_run = AverageMeter('pole', ':.4f')

        self.traffic_sign1_run = AverageMeter('traiffic_sign', ':.4f')
        self.traffic_sign2_run = AverageMeter('traiffic_sign', ':.4f')
        self.traffic_sign3_run = AverageMeter('traiffic_sign', ':.4f')
        self.traffic_sign4_run = AverageMeter('traiffic_sign', ':.4f')
        self.traffic_sign5_run = AverageMeter('traiffic_sign', ':.4f')
        self.traffic_sign6_run = AverageMeter('traiffic_sign', ':.4f')

        self.loss1_run.reset()
        self.loss2_run.reset()
        self.miou1_run.reset()
        self.miou2_run.reset()
        self.miou3_run.reset()
        self.miou4_run.reset()
        self.miou5_run.reset()
        self.miou6_run.reset()

        self.car1_run.reset()
        self.car2_run.reset()
        self.car3_run.reset()
        self.car4_run.reset()
        self.car5_run.reset()
        self.car6_run.reset()

        self.bicycle1_run.reset()
        self.bicycle2_run.reset()
        self.bicycle3_run.reset()
        self.bicycle4_run.reset()
        self.bicycle5_run.reset()
        self.bicycle6_run.reset()

        self.motorcycle1_run.reset()
        self.motorcycle2_run.reset()
        self.motorcycle3_run.reset()
        self.motorcycle4_run.reset()
        self.motorcycle5_run.reset()
        self.motorcycle6_run.reset()

        self.truck1_run.reset()
        self.truck2_run.reset()
        self.truck3_run.reset()
        self.truck4_run.reset()
        self.truck5_run.reset()
        self.truck6_run.reset()

        self.other_vehicle1_run.reset()
        self.other_vehicle2_run.reset()
        self.other_vehicle3_run.reset()
        self.other_vehicle4_run.reset()
        self.other_vehicle5_run.reset()
        self.other_vehicle6_run.reset()

        self.person1_run.reset()
        self.person2_run.reset()
        self.person3_run.reset()
        self.person4_run.reset()
        self.person5_run.reset()
        self.person6_run.reset()

        self.bicyclist1_run.reset()
        self.bicyclist2_run.reset()
        self.bicyclist3_run.reset()
        self.bicyclist4_run.reset()
        self.bicyclist5_run.reset()
        self.bicyclist6_run.reset()

        self.motorcyclist1_run.reset()
        self.motorcyclist2_run.reset()
        self.motorcyclist3_run.reset()
        self.motorcyclist4_run.reset()
        self.motorcyclist5_run.reset()
        self.motorcyclist6_run.reset()

        self.road1_run.reset()
        self.road2_run.reset()
        self.road3_run.reset()
        self.road4_run.reset()
        self.road5_run.reset()
        self.road6_run.reset()

        self.parking1_run.reset()
        self.parking2_run.reset()
        self.parking3_run.reset()
        self.parking4_run.reset()
        self.parking5_run.reset()
        self.parking6_run.reset()

        self.sidewalk1_run.reset()
        self.sidewalk2_run.reset()
        self.sidewalk3_run.reset()
        self.sidewalk4_run.reset()
        self.sidewalk5_run.reset()
        self.sidewalk6_run.reset()

        self.other_ground1_run.reset()
        self.other_ground2_run.reset()
        self.other_ground3_run.reset()
        self.other_ground4_run.reset()
        self.other_ground5_run.reset()
        self.other_ground6_run.reset()

        self.building1_run.reset()
        self.building2_run.reset()
        self.building3_run.reset()
        self.building4_run.reset()
        self.building5_run.reset()
        self.building6_run.reset()

        self.fence1_run.reset()
        self.fence2_run.reset()
        self.fence3_run.reset()
        self.fence4_run.reset()
        self.fence5_run.reset()
        self.fence6_run.reset()

        self.vegetation1_run.reset()
        self.vegetation2_run.reset()
        self.vegetation3_run.reset()
        self.vegetation4_run.reset()
        self.vegetation5_run.reset()
        self.vegetation6_run.reset()

        self.trunk1_run.reset()
        self.trunk2_run.reset()
        self.trunk3_run.reset()
        self.trunk4_run.reset()
        self.trunk5_run.reset()
        self.trunk6_run.reset()

        self.terrain1_run.reset()
        self.terrain2_run.reset()
        self.terrain3_run.reset()
        self.terrain4_run.reset()
        self.terrain5_run.reset()
        self.terrain6_run.reset()

        self.pole1_run.reset()
        self.pole2_run.reset()
        self.pole3_run.reset()
        self.pole4_run.reset()
        self.pole5_run.reset()
        self.pole6_run.reset()

        self.traffic_sign1_run.reset()
        self.traffic_sign2_run.reset()
        self.traffic_sign3_run.reset()
        self.traffic_sign4_run.reset()
        self.traffic_sign5_run.reset()
        self.traffic_sign6_run.reset()

        self.iou1 = IOUEval(n_classes=20, ignore=0)
        self.iou2 = IOUEval(n_classes=20, ignore=0)
        self.iou_3d_label = IOUEval(n_classes=20, ignore=0)
        self.iou_3d_label_3d = IOUEval(n_classes=20, ignore=0)
        self.iou_3d_knn_label = IOUEval(n_classes=20, ignore=0)
        self.iou_3d_knn_label_3d = IOUEval(n_classes=20, ignore=0)

    def set_dir(self):
        if 'earlystop' in self.ckpt_path :
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')), 'test_result','side', f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_knn{self.knn.knn}_search{self.knn.search}")
        elif 'miou' in self.ckpt_path :
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')), 'test_result','side', f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_miou{self.miou}_knn{self.knn.knn}_search{self.knn.search}")
        elif 'last' in self.ckpt_path:
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')), 'test_result','side', f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_miou{self.miou}_knn{self.knn.knn}_search{self.knn.search}")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_loader(self):
        self.data_path = 'data/semantic_kitti/kitti/dataset/sequences'

        dataset = SemanticKITTI(self.data_path, shape=(384, 1248), nclasses=20, 
                                mode='valid', front=360, split=False, crop_size=128, thetas=self.thetas)
                                # mode='valid', front=360, split=False, crop_size=128, thetas=[161])
        self.loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)

        cfg_path = 'data_loader/semantic-kitti.yaml'
        CFG = yaml.safe_load(open(cfg_path, 'r'))
        color_dict = CFG['color_map']
        learning_map = CFG['learning_map']
        self.learning_map_inv = CFG['learning_map_inv']
        self.color_dict = {learning_map[key]:color_dict[self.learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

    def convert_color(self, arr):
        result = np.zeros((*arr.shape, 3))
        for b in range(result.shape[0]):
            for i in self.color_dict:
                j = np.where(arr[b]==i)
                try:
                    xs, ys = j[0], j[1]
                except:
                    xs = j[0]

                if len(xs) == 0:
                    continue
                bgr = self.color_dict[i]
                for x, y in zip(xs, ys):
                    result[b, x, y] = bgr
        return result

    def logging2(self):
        loss1, loss2, miou1, miou2 = self.loss1_run.avg, self.loss2_run.avg, self.miou1_run.avg, self.miou2_run.avg
        miou3, miou4, miou5, miou6 = self.miou3_run.avg, self.miou4_run.avg, self.miou5_run.avg, self.miou6_run.avg

        car1, bicycle1, motorcycle1, truck1, other_vehicle1 = self.car1_run.avg, self.bicycle1_run.avg, self.motorcycle1_run.avg, self.truck1_run.avg, self.other_vehicle1_run.avg
        car2, bicycle2, motorcycle2, truck2, other_vehicle2 = self.car2_run.avg, self.bicycle2_run.avg, self.motorcycle2_run.avg, self.truck2_run.avg, self.other_vehicle2_run.avg
        car3, bicycle3, motorcycle3, truck3, other_vehicle3 = self.car3_run.avg, self.bicycle3_run.avg, self.motorcycle3_run.avg, self.truck3_run.avg, self.other_vehicle3_run.avg
        car4, bicycle4, motorcycle4, truck4, other_vehicle4 = self.car4_run.avg, self.bicycle4_run.avg, self.motorcycle4_run.avg, self.truck4_run.avg, self.other_vehicle4_run.avg
        car5, bicycle5, motorcycle5, truck5, other_vehicle5 = self.car5_run.avg, self.bicycle5_run.avg, self.motorcycle5_run.avg, self.truck5_run.avg, self.other_vehicle5_run.avg
        car6, bicycle6, motorcycle6, truck6, other_vehicle6 = self.car6_run.avg, self.bicycle6_run.avg, self.motorcycle6_run.avg, self.truck6_run.avg, self.other_vehicle6_run.avg
        
        person1, bicyclist1, motorcyclist1, road1, parking1 = self.person1_run.avg, self.bicyclist1_run.avg, self.motorcyclist1_run.avg, self.road1_run.avg, self.parking1_run.avg
        person2, bicyclist2, motorcyclist2, road2, parking2 = self.person2_run.avg, self.bicyclist2_run.avg, self.motorcyclist2_run.avg, self.road2_run.avg, self.parking2_run.avg
        person3, bicyclist3, motorcyclist3, road3, parking3 = self.person3_run.avg, self.bicyclist3_run.avg, self.motorcyclist3_run.avg, self.road3_run.avg, self.parking3_run.avg
        person4, bicyclist4, motorcyclist4, road4, parking4 = self.person4_run.avg, self.bicyclist4_run.avg, self.motorcyclist4_run.avg, self.road4_run.avg, self.parking4_run.avg
        person5, bicyclist5, motorcyclist5, road5, parking5 = self.person5_run.avg, self.bicyclist5_run.avg, self.motorcyclist5_run.avg, self.road5_run.avg, self.parking5_run.avg
        person6, bicyclist6, motorcyclist6, road6, parking6 = self.person6_run.avg, self.bicyclist6_run.avg, self.motorcyclist6_run.avg, self.road6_run.avg, self.parking6_run.avg

        sidewalk1, other_ground1, building1, fence1, vegetation1 = self.sidewalk1_run.avg, self.other_ground1_run.avg, self.building1_run.avg, self.fence1_run.avg, self.vegetation1_run.avg
        sidewalk2, other_ground2, building2, fence2, vegetation2 = self.sidewalk2_run.avg, self.other_ground2_run.avg, self.building2_run.avg, self.fence2_run.avg, self.vegetation2_run.avg
        sidewalk3, other_ground3, building3, fence3, vegetation3 = self.sidewalk3_run.avg, self.other_ground3_run.avg, self.building3_run.avg, self.fence3_run.avg, self.vegetation3_run.avg
        sidewalk4, other_ground4, building4, fence4, vegetation4 = self.sidewalk4_run.avg, self.other_ground4_run.avg, self.building4_run.avg, self.fence4_run.avg, self.vegetation4_run.avg
        sidewalk5, other_ground5, building5, fence5, vegetation5 = self.sidewalk5_run.avg, self.other_ground5_run.avg, self.building5_run.avg, self.fence5_run.avg, self.vegetation5_run.avg
        sidewalk6, other_ground6, building6, fence6, vegetation6 = self.sidewalk6_run.avg, self.other_ground6_run.avg, self.building6_run.avg, self.fence6_run.avg, self.vegetation6_run.avg

        trunk1, terrain1, pole1, traffic_sign1 = self.trunk1_run.avg, self.terrain1_run.avg, self.pole1_run.avg, self.traffic_sign1_run.avg
        trunk2, terrain2, pole2, traffic_sign2 = self.trunk2_run.avg, self.terrain2_run.avg, self.pole2_run.avg, self.traffic_sign2_run.avg
        trunk3, terrain3, pole3, traffic_sign3 = self.trunk3_run.avg, self.terrain3_run.avg, self.pole3_run.avg, self.traffic_sign3_run.avg
        trunk4, terrain4, pole4, traffic_sign4 = self.trunk4_run.avg, self.terrain4_run.avg, self.pole4_run.avg, self.traffic_sign4_run.avg
        trunk5, terrain5, pole5, traffic_sign5 = self.trunk5_run.avg, self.terrain5_run.avg, self.pole5_run.avg, self.traffic_sign5_run.avg
        trunk6, terrain6, pole6, traffic_sign6 = self.trunk6_run.avg, self.terrain6_run.avg, self.pole6_run.avg, self.traffic_sign6_run.avg

        print('\n[Test] | Loss1 {:.4f} | Loss2 {:.4f} | mIoU1 {:.4f} | mIoU2 {:.4f} | mIoU 3D {:.4f}'.format(loss1, loss2, miou1, miou2, miou3))
        print(f"\n[Test] | Loss1 {loss1:.4f} | Loss2 {loss2:.4f} | mIoU1 {miou1:.4f} |mIoU2 {miou2:.4f}")
        print(f"\t\t\t\tlabel outsegC {miou3:.4f} | label_c outsegC {miou4:.4f} | label knnoutC {miou5:.4f} | label_c knnoutC {miou6:.4f}")

        print("mIoU Reuslt", "*"*25)
        print(f'\nmIoU1\t\t{miou1:.4f}\t\tLoss1\t{loss1:.4f}')
        print(f'\nmIoU2\t\t{miou2:.4f}\t\tLoss2\t{loss2:.4f}')
        print(f'\nlabel outsegC\t\t{miou3:.4f}')
        print(f'\nlabel_c outsegC\t\t{miou4:.4f}')
        print(f'\nlabel knnoutC\t\t{miou5:.4f}')
        print(f'\nlabel_c knnoutC\t\t{miou6:.4f}')

        print("IoU per class \n", "*"*25)
        print(f"label   pred    car     bicycle     motorcycle      truck       other_vehicle       person      bicyclist       motorcyclist        road        parking     sidewalk        other_ground     building        fence       vegetation      trunk       terrain     pole        traffic_sign")
        print(f"proj    seg1    {car1:.4f}      {bicycle1:.4f}      {motorcycle1:.4f}       {truck1:.4f}        {other_vehicle1:.4f}        {person1:.4f}       {bicyclist1:.4f}        {motorcyclist1:.4f}     {road1:.4f}     {parking1:.4f}      {sidewalk1:.4f}     {other_ground1:.4f}     {building1:.4f}     {fence1:.4f}        {vegetation1:.4f}       {trunk1:.4f}        {terrain1:.4f}      {pole1:.4f}     {traffic_sign1:.4f}")
        print(f"proj    seg2    {car2:.4f}      {bicycle2:.4f}      {motorcycle2:.4f}       {truck2:.4f}        {other_vehicle2:.4f}        {person2:.4f}       {bicyclist2:.4f}        {motorcyclist2:.4f}     {road2:.4f}     {parking2:.4f}      {sidewalk2:.4f}     {other_ground2:.4f}     {building2:.4f}     {fence2:.4f}        {vegetation2:.4f}       {trunk2:.4f}        {terrain2:.4f}      {pole2:.4f}     {traffic_sign2:.4f}")
        print(f"label   seg2C   {car3:.4f}      {bicycle3:.4f}      {motorcycle3:.4f}       {truck3:.4f}        {other_vehicle3:.4f}        {person3:.4f}       {bicyclist3:.4f}        {motorcyclist3:.4f}     {road3:.4f}     {parking3:.4f}      {sidewalk3:.4f}     {other_ground3:.4f}     {building3:.4f}     {fence3:.4f}        {vegetation3:.4f}       {trunk3:.4f}        {terrain3:.4f}      {pole3:.4f}     {traffic_sign3:.4f}")
        print(f"label_c seg2C   {car4:.4f}      {bicycle4:.4f}      {motorcycle4:.4f}       {truck4:.4f}        {other_vehicle4:.4f}        {person4:.4f}       {bicyclist4:.4f}        {motorcyclist4:.4f}     {road4:.4f}     {parking4:.4f}      {sidewalk4:.4f}     {other_ground4:.4f}     {building4:.4f}     {fence4:.4f}        {vegetation4:.4f}       {trunk4:.4f}        {terrain4:.4f}      {pole4:.4f}     {traffic_sign4:.4f}")
        print(f"label   knnC    {car5:.4f}      {bicycle5:.4f}      {motorcycle5:.4f}       {truck5:.4f}        {other_vehicle5:.4f}        {person5:.4f}       {bicyclist5:.4f}        {motorcyclist5:.4f}     {road5:.4f}     {parking5:.4f}      {sidewalk5:.4f}     {other_ground5:.4f}     {building5:.4f}     {fence5:.4f}        {vegetation5:.4f}       {trunk5:.4f}        {terrain5:.4f}      {pole5:.4f}     {traffic_sign5:.4f}")
        print(f"label_c knnC    {car6:.4f}      {bicycle6:.4f}      {motorcycle6:.4f}       {truck6:.4f}        {other_vehicle6:.4f}        {person6:.4f}       {bicyclist6:.4f}        {motorcyclist6:.4f}     {road6:.4f}     {parking6:.4f}      {sidewalk6:.4f}     {other_ground6:.4f}     {building6:.4f}     {fence6:.4f}        {vegetation6:.4f}       {trunk6:.4f}        {terrain6:.4f}      {pole6:.4f}     {traffic_sign6:.4f}")

        with open(f'{self.save_dir}/result.txt','w') as f:
            f.write("\nmIoU Reuslt -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            f.write(f'\nmIoU1\t\t{miou1:.4f}\t\tLoss1\t{loss1:.4f}')
            f.write(f'\nmIoU2\t\t{miou2:.4f}\t\tLoss2\t{loss2:.4f}')
            f.write(f'\nlabel outsegC\t\t{miou3:.4f}')
            f.write(f'\nlabel_c outsegC\t\t{miou4:.4f}')
            f.write(f'\nlabel knnoutC\t\t{miou5:.4f}')
            f.write(f'\nlabel_c knnoutC\t\t{miou6:.4f}')
            f.write("\nIoU per class-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            f.write(f"\nlabel   pred    car     bicycle     motorcycle      truck       other_vehicle       person      bicyclist       motorcyclist        road        parking     sidewalk        other_ground     building        fence       vegetation      trunk       terrain     pole        traffic_sign")
            f.write(f"\nproj    seg1    {car1:.4f}      {bicycle1:.4f}      {motorcycle1:.4f}       {trunk1:.4f}        {other_vehicle1:.4f}        {person1:.4f}       {bicyclist1:.4f}        {motorcyclist1:.4f}     {road1:.4f}     {parking1:.4f}      {sidewalk1:.4f}     {other_ground1:.4f}     {building1:.4f}     {fence1:.4f}        {vegetation1:.4f}       {trunk1:.4f}        {terrain1:.4f}      {pole1:.4f}     {traffic_sign1:.4f}")
            f.write(f"\nproj    seg2    {car2:.4f}      {bicycle2:.4f}      {motorcycle2:.4f}       {trunk2:.4f}        {other_vehicle2:.4f}        {person2:.4f}       {bicyclist2:.4f}        {motorcyclist2:.4f}     {road2:.4f}     {parking2:.4f}      {sidewalk2:.4f}     {other_ground2:.4f}     {building2:.4f}     {fence2:.4f}        {vegetation2:.4f}       {trunk2:.4f}        {terrain2:.4f}      {pole2:.4f}     {traffic_sign2:.4f}")
            f.write(f"\nlabel   seg2C   {car3:.4f}      {bicycle3:.4f}      {motorcycle3:.4f}       {trunk3:.4f}        {other_vehicle3:.4f}        {person3:.4f}       {bicyclist3:.4f}        {motorcyclist3:.4f}     {road3:.4f}     {parking3:.4f}      {sidewalk3:.4f}     {other_ground3:.4f}     {building3:.4f}     {fence3:.4f}        {vegetation3:.4f}       {trunk3:.4f}        {terrain3:.4f}      {pole3:.4f}     {traffic_sign3:.4f}")
            f.write(f"\nlabel_c seg2C   {car4:.4f}      {bicycle4:.4f}      {motorcycle4:.4f}       {trunk4:.4f}        {other_vehicle4:.4f}        {person4:.4f}       {bicyclist4:.4f}        {motorcyclist4:.4f}     {road4:.4f}     {parking4:.4f}      {sidewalk4:.4f}     {other_ground4:.4f}     {building4:.4f}     {fence4:.4f}        {vegetation4:.4f}       {trunk4:.4f}        {terrain4:.4f}      {pole4:.4f}     {traffic_sign4:.4f}")
            f.write(f"\nlabel   knnC    {car5:.4f}      {bicycle5:.4f}      {motorcycle5:.4f}       {trunk5:.4f}        {other_vehicle5:.4f}        {person5:.4f}       {bicyclist5:.4f}        {motorcyclist5:.4f}     {road5:.4f}     {parking5:.4f}      {sidewalk5:.4f}     {other_ground5:.4f}     {building5:.4f}     {fence5:.4f}        {vegetation5:.4f}       {trunk5:.4f}        {terrain5:.4f}      {pole5:.4f}     {traffic_sign5:.4f}")
            f.write(f"\nlabel_c knnC    {car6:.4f}      {bicycle6:.4f}      {motorcycle6:.4f}       {trunk6:.4f}        {other_vehicle6:.4f}        {person6:.4f}       {bicyclist6:.4f}        {motorcyclist6:.4f}     {road6:.4f}     {parking6:.4f}      {sidewalk6:.4f}     {other_ground6:.4f}     {building6:.4f}     {fence6:.4f}        {vegetation6:.4f}       {trunk6:.4f}        {terrain6:.4f}      {pole6:.4f}     {traffic_sign6:.4f}")
        f.close()

    def vis_3d(self, outs_seg1, outs_seg2, proj_range, unproj_range, px, py, x, y):
        out1_np = outs_seg1.detach().cpu().numpy()
        out2_np = outs_seg2.detach().cpu().numpy()
        proj_range_np = proj_range[0].detach().cpu().numpy()
        unproj_range_np = unproj_range.detach().cpu().numpy()
        px_np = px.detach().cpu().numpy()
        py_np = py.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        print(out1_np.shape)
        print(out2_np.shape)
        print(proj_range_np.shape)
        print(unproj_range_np.shape)
        print(px_np.shape)
        print(x_np.shape)
        print(y_np.shape)

        np.save("out1_np.npy", out1_np)
        np.save("out2_np.npy", out2_np)
        np.save("proj_range_np.npy", proj_range_np)
        np.save("unproj_range_np.npy", unproj_range_np)
        np.save("px_np.npy", px_np)
        np.save("py_np.npy", py_np)
        np.save("x_np.npy", x_np)
        np.save("y_np.npy", y_np)
        exit()
    
    def logging(self, bs):
        self.miou1_run.update(self.miou1, bs)
        self.miou2_run.update(self.miou2, bs)
        self.miou3_run.update(self.miou3, bs)
        self.miou4_run.update(self.miou4, bs)
        self.miou5_run.update(self.miou5, bs)
        self.miou6_run.update(self.miou6, bs)

        self.loss1_run.update(self.loss1.item(), bs)
        self.loss2_run.update(self.loss2.item(), bs)

        self.car1_run.update(self.per_iou1[1].item(), bs)
        self.car2_run.update(self.per_iou2[1].item(), bs)
        self.car3_run.update(self.per_iou3[1].item(), bs)
        self.car4_run.update(self.per_iou4[1].item(), bs)
        self.car5_run.update(self.per_iou5[1].item(), bs)
        self.car6_run.update(self.per_iou6[1].item(), bs)

        self.bicycle1_run.update(self.per_iou1[2].item(), bs)
        self.bicycle2_run.update(self.per_iou2[2].item(), bs)
        self.bicycle3_run.update(self.per_iou3[2].item(), bs)
        self.bicycle4_run.update(self.per_iou4[2].item(), bs)
        self.bicycle5_run.update(self.per_iou5[2].item(), bs)
        self.bicycle6_run.update(self.per_iou6[2].item(), bs)

        self.motorcycle1_run.update(self.per_iou1[3].item(), bs)
        self.motorcycle2_run.update(self.per_iou2[3].item(), bs)
        self.motorcycle3_run.update(self.per_iou3[3].item(), bs)
        self.motorcycle4_run.update(self.per_iou4[3].item(), bs)
        self.motorcycle5_run.update(self.per_iou5[3].item(), bs)
        self.motorcycle6_run.update(self.per_iou6[3].item(), bs)

        self.truck1_run.update(self.per_iou1[4].item(), bs)
        self.truck2_run.update(self.per_iou2[4].item(), bs)
        self.truck3_run.update(self.per_iou3[4].item(), bs)
        self.truck4_run.update(self.per_iou4[4].item(), bs)
        self.truck5_run.update(self.per_iou5[4].item(), bs)
        self.truck6_run.update(self.per_iou6[4].item(), bs)

        self.other_vehicle1_run.update(self.per_iou1[5].item(), bs)
        self.other_vehicle2_run.update(self.per_iou2[5].item(), bs)
        self.other_vehicle3_run.update(self.per_iou3[5].item(), bs)
        self.other_vehicle4_run.update(self.per_iou4[5].item(), bs)
        self.other_vehicle5_run.update(self.per_iou5[5].item(), bs)
        self.other_vehicle6_run.update(self.per_iou6[5].item(), bs)

        self.person1_run.update(self.per_iou1[6].item(), bs)
        self.person2_run.update(self.per_iou2[6].item(), bs)
        self.person3_run.update(self.per_iou3[6].item(), bs)
        self.person4_run.update(self.per_iou4[6].item(), bs)
        self.person5_run.update(self.per_iou5[6].item(), bs)
        self.person6_run.update(self.per_iou6[6].item(), bs)

        self.bicyclist1_run.update(self.per_iou1[7].item(), bs)
        self.bicyclist2_run.update(self.per_iou2[7].item(), bs)
        self.bicyclist3_run.update(self.per_iou3[7].item(), bs)
        self.bicyclist4_run.update(self.per_iou4[7].item(), bs)
        self.bicyclist5_run.update(self.per_iou5[7].item(), bs)
        self.bicyclist6_run.update(self.per_iou6[7].item(), bs)

        self.motorcyclist1_run.update(self.per_iou1[8].item(), bs)
        self.motorcyclist2_run.update(self.per_iou2[8].item(), bs)
        self.motorcyclist3_run.update(self.per_iou3[8].item(), bs)
        self.motorcyclist4_run.update(self.per_iou4[8].item(), bs)
        self.motorcyclist5_run.update(self.per_iou5[8].item(), bs)
        self.motorcyclist6_run.update(self.per_iou6[8].item(), bs)

        self.road1_run.update(self.per_iou1[9].item(), bs)
        self.road2_run.update(self.per_iou2[9].item(), bs)
        self.road3_run.update(self.per_iou3[9].item(), bs)
        self.road4_run.update(self.per_iou4[9].item(), bs)
        self.road5_run.update(self.per_iou5[9].item(), bs)
        self.road6_run.update(self.per_iou6[9].item(), bs)
        
        self.parking1_run.update(self.per_iou1[10].item(), bs)
        self.parking2_run.update(self.per_iou2[10].item(), bs)
        self.parking3_run.update(self.per_iou3[10].item(), bs)
        self.parking4_run.update(self.per_iou4[10].item(), bs)
        self.parking5_run.update(self.per_iou5[10].item(), bs)
        self.parking6_run.update(self.per_iou6[10].item(), bs)

        self.sidewalk1_run.update(self.per_iou1[11].item(), bs)
        self.sidewalk2_run.update(self.per_iou2[11].item(), bs)
        self.sidewalk3_run.update(self.per_iou3[11].item(), bs)
        self.sidewalk4_run.update(self.per_iou4[11].item(), bs)
        self.sidewalk5_run.update(self.per_iou5[11].item(), bs)
        self.sidewalk6_run.update(self.per_iou6[11].item(), bs)

        self.other_ground1_run.update(self.per_iou1[12].item(), bs)
        self.other_ground2_run.update(self.per_iou2[12].item(), bs)
        self.other_ground3_run.update(self.per_iou3[12].item(), bs)
        self.other_ground4_run.update(self.per_iou4[12].item(), bs)
        self.other_ground5_run.update(self.per_iou5[12].item(), bs)
        self.other_ground6_run.update(self.per_iou6[12].item(), bs)

        self.building1_run.update(self.per_iou1[13].item(), bs)
        self.building2_run.update(self.per_iou2[13].item(), bs)
        self.building3_run.update(self.per_iou3[13].item(), bs)
        self.building4_run.update(self.per_iou4[13].item(), bs)
        self.building5_run.update(self.per_iou5[13].item(), bs)
        self.building6_run.update(self.per_iou6[13].item(), bs)

        self.fence1_run.update(self.per_iou1[14].item(), bs)
        self.fence2_run.update(self.per_iou2[14].item(), bs)
        self.fence3_run.update(self.per_iou3[14].item(), bs)
        self.fence4_run.update(self.per_iou4[14].item(), bs)
        self.fence5_run.update(self.per_iou5[14].item(), bs)
        self.fence6_run.update(self.per_iou6[14].item(), bs)

        self.vegetation1_run.update(self.per_iou1[15].item(), bs)
        self.vegetation2_run.update(self.per_iou2[15].item(), bs)
        self.vegetation3_run.update(self.per_iou3[15].item(), bs)
        self.vegetation4_run.update(self.per_iou4[15].item(), bs)
        self.vegetation5_run.update(self.per_iou5[15].item(), bs)
        self.vegetation6_run.update(self.per_iou6[15].item(), bs)

        self.trunk1_run.update(self.per_iou1[16].item(), bs)
        self.trunk2_run.update(self.per_iou2[16].item(), bs)
        self.trunk3_run.update(self.per_iou3[16].item(), bs)
        self.trunk4_run.update(self.per_iou4[16].item(), bs)
        self.trunk5_run.update(self.per_iou5[16].item(), bs)
        self.trunk6_run.update(self.per_iou6[16].item(), bs)

        self.terrain1_run.update(self.per_iou1[17].item(), bs)
        self.terrain2_run.update(self.per_iou2[17].item(), bs)
        self.terrain3_run.update(self.per_iou3[17].item(), bs)
        self.terrain4_run.update(self.per_iou4[17].item(), bs)
        self.terrain5_run.update(self.per_iou5[17].item(), bs)
        self.terrain6_run.update(self.per_iou6[17].item(), bs)

        self.pole1_run.update(self.per_iou1[18].item(), bs)
        self.pole2_run.update(self.per_iou2[18].item(), bs)
        self.pole3_run.update(self.per_iou3[18].item(), bs)
        self.pole4_run.update(self.per_iou4[18].item(), bs)
        self.pole5_run.update(self.per_iou5[18].item(), bs)
        self.pole6_run.update(self.per_iou6[18].item(), bs)

        self.traffic_sign1_run.update(self.per_iou1[19].item(), bs)
        self.traffic_sign2_run.update(self.per_iou2[19].item(), bs)
        self.traffic_sign3_run.update(self.per_iou3[19].item(), bs)
        self.traffic_sign4_run.update(self.per_iou4[19].item(), bs)
        self.traffic_sign5_run.update(self.per_iou5[19].item(), bs)
        self.traffic_sign6_run.update(self.per_iou6[19].item(), bs)

    def inference(self):
        ckpts = os.listdir(self.segmentor_path)   
        ckpts = [os.path.join(self.segmentor_path, ckpt) for ckpt in ckpts if 'pt' in ckpt]     

        for idx, ckpt in enumerate(ckpts):
            self.ckpt_path = ckpt
            self.set_init()
            self.set_model()
            self.set_dir()
            progress = ProgressMeter(len(self.loader), 
                                     [self.miou3_run, self.miou4_run, self.miou5_run, self.miou6_run],
                                     prefix=f"{idx+1}/{len(ckpts)} {self.mode}")
            
            self.iou1.reset()
            self.iou2.reset()
            self.iou_3d_label.reset()
            self.iou_3d_label_3d.reset()
            self.iou_3d_knn_label.reset()
            self.iou_3d_knn_label_3d.reset()

            self.generator.eval()
            self.segmentor.eval()
            with torch.no_grad():
                for iter, batch in enumerate(self.loader):
                    
                    inputs_rdm = batch['rdm'].to(self.device).squeeze(0)[:,:,self.crop_size:,]
                    labels = batch['label'].to(self.device).squeeze(0)[:,self.crop_size:,]
                    
                    proj_ranges = batch['proj_range'].squeeze(0).to(self.device)
                    npoints = batch['npoints'].squeeze(0).to(self.device)
                    unproj_ranges = batch['unproj_range'].squeeze(0).to(self.device)
                    pxs, pys = batch['px'].squeeze(0).to(self.device), batch['py'].squeeze(0).to(self.device)
                    labels_3d = batch['label_3d'].squeeze(0).to(self.device)
                    
                    bs = inputs_rdm.size(0)
                    self.loss1 = 0
                    self.loss2 = 0
                    for b in range(bs):
                        input_rdm = inputs_rdm[b].unsqueeze(0)
                        label = labels[b].unsqueeze(0)
                        proj_range = proj_ranges[b]
                        npoint = int(npoints[b].item())
                        label_3d = labels_3d[b][:npoint]
                        unproj_range = unproj_ranges[b][:npoint]
                        px, py = pxs[b][:npoint], pys[b][:npoint]

                        out_rgb, out_seg1 = self.generator(input_rdm)
                        out_seg2 = self.segmentor(out_rgb, input_rdm)

                        self.loss1 += self.criterion(out_seg1, label).detach()
                        self.loss2 += self.criterion(out_seg2, label).detach()

                        # semantic kitti 360 knn
                        out_seg2_temp = torch.cat([torch.zeros(self.crop_size, 1248).to(self.device), torch.argmax(out_seg2[0], 0)], dim=0)
                        out_knn = self.knn(proj_range, unproj_range, out_seg2_temp, px, py)[:npoint]

                        py = py.detach().cpu().numpy()[:npoint]
                        px = px.detach().cpu().numpy()[:npoint]

                        label_temp = np.zeros([self.crop_size, 1248])
                        label_temp = np.concatenate([label_temp, label[0].detach().cpu().numpy()], 0)

                        # 'label'을 이용한 2d seg map
                        label_c = np.zeros(px.shape[-1])
                        for i, (h, w) in enumerate(zip(py, px)):
                            label_c[i] = int(label_temp[int(h), int(w)])
                        label_seg = np.zeros([384, 1248])
                        for h, w, c in zip(py, px, label_c):
                            label_seg[int(h), int(w)] = int(c)
                        label_seg = self.convert_color(np.expand_dims(label_seg,0))[0]

                        # label 3d(point class)로 2D seg map 
                        label_3d_seg = np.zeros([384, 1248])
                        for h, w, c in zip(py, px, label_3d.detach().cpu().numpy()):
                            label_3d_seg[int(h), int(w)] = int(c)
                        label_3d_seg = self.convert_color(np.expand_dims(label_3d_seg[self.crop_size:,], 0))[0]

                        # knn output으로 2D seg map 생성 
                        out_knn_seg = np.zeros([384, 1248])
                        for h, w, c in zip(py, px, out_knn.detach().cpu().numpy()):
                            out_knn_seg[int(h), int(w)] = int(c)
                        out_knn_seg = self.convert_color(np.expand_dims(out_knn_seg[self.crop_size:,], 0))[0]

                        # seg out c h w -> n
                        out_seg2_c = np.zeros(px.shape[0])
                        for i, (h, w) in enumerate(zip(py, px)):
                            out_seg2_c[i] = out_seg2_temp[int(h), int(w)].item()
                        # c = rearrange(batch['label_c'][0], 'c h w -> h w c')[self.crop_size:,:,].numpy()*255
                        # label_rgb = rearrange(batch['img'][0], 'c h w -> h w c').detach().cpu().numpy()
                        # cv2.imwrite(f'{self.save_dir}/test3.png', np.concatenate([outs_org, label_3d_seg, c, label_rgb], axis=0))

                        # a2 = np.where(outs_org[self.crop_size:,]>0, outs_org[self.crop_size:], label_rgb)
                        # b2 = np.where(label_3d_seg>0, label_3d_seg, label_rgb)
                        # c2 = np.where(c>0, c, label_rgb)
                        # cv2.imwrite(f'{self.save_dir}/test4.png', np.concatenate([a2, b2, c2], axis=0))
                        out_knn = out_knn.detach().cpu().numpy()
                        label_3d = label_3d.detach().cpu().numpy()

                        self.iou1.addBatch(torch.argmax(out_seg1, 1), label)
                        self.iou2.addBatch(torch.argmax(out_seg2, 1), label)
                        self.iou_3d_label.addBatch(out_seg2_c, label_c)
                        self.iou_3d_knn_label.addBatch(out_knn, label_c)
                        self.iou_3d_label_3d.addBatch(out_seg2_c, label_3d)
                        self.iou_3d_knn_label_3d.addBatch(out_knn, label_3d)
                    self.loss1 /= len(self.thetas)
                    self.loss2 /= len(self.thetas)

                    self.miou1, self.per_iou1 = self.iou1.getIoU()
                    self.miou2, self.per_iou2 = self.iou2.getIoU()
                    self.miou3, self.per_iou3 = self.iou_3d_label.getIoU()
                    self.miou4, self.per_iou4 = self.iou_3d_knn_label.getIoU()
                    self.miou5, self.per_iou5 = self.iou_3d_label_3d.getIoU()
                    self.miou6, self.per_iou6 = self.iou_3d_knn_label_3d.getIoU()
                    
                    self.logging(bs)
                    
                    progress.display(iter)

                    # outs_seg1 = self.convert_color(torch.argmax(outs_seg1, 1).cpu().detach().numpy())[0] # b h w c
                    # outs_seg2 = self.convert_color(torch.argmax(outs_seg2, 1).cpu().detach().numpy())[0]
                    # inputs_rdm= rearrange(inputs_rdm, 'b c h w -> b h w c').cpu().detach().numpy()[0]
                    # outs_rgb = rearrange(outs_rgb, 'b c h w -> b h w c').cpu().detach().numpy()[0]*255
                    # labels_c = rearrange(batch['label_c'][0], 'c h w -> h w c').numpy()*255


                    # cv2.imwrite(f"{self.save_dir}/vis_outseg.png", np.concatenate([labels_c, outs_seg1, outs_seg2, inputs_rdm], axis=0))
                        
                    # for i in range(5):
                    #     result = np.concatenate([inputs_rdm[i], outs_seg1[i], outs_rgb[i], outs_seg2[i], labels_c[i]], axis=-2)
                    #     cv2.imwrite(f"{self.save_dir}/{str(i)}/{str(iter)}.png", result)

                    # cv2.imwrite(f"{self.save_dir}/0/segment_0.png", self.convert_color(torch.argmax(outs_seg, 1)[0][0].cpu().detach().numpy(), self.color_dict))
                    # cv2.imwrite(f"{self.save_dir}/sample/segment_1.png", self.convert_color(torch.argmax(outs_seg, 1)[0][1].cpu().detach().numpy(), self.color_dict))
                    # cv2.imwrite(f"{self.save_dir}/sample/segment_2.png", self.convert_color(torch.argmax(outs_seg, 1)[0][2].cpu().detach().numpy(), self.color_dict))
                    # cv2.imwrite(f"{self.save_dir}/sample/segment_3.png", self.convert_color(torch.argmax(outs_seg, 1)[0][3].cpu().detach().numpy(), self.color_dict))
                    # cv2.imwrite(f"{self.save_dir}/sample/segment_4.png", self.convert_color(torch.argmax(outs_seg, 1)[0][4].cpu().detach().numpy(), self.color_dict))
                self.logging2()

if __name__ == "__main__":
    Inference().inference()