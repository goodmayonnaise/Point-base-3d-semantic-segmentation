
import cv2, os, time, yaml
import numpy as np
from einops import rearrange
from datetime import timedelta

from utils.metrics import IOUEval
from utils.logs import AverageMeter, ProgressMeter
from losses.focal_lovasz import FocalLosswithLovaszRegularizer
from dataloader.semantic_kitti_360 import SemanticKITTI
from models.segmentor import EncoderDecoder as Segmentor
from models.generator import EncoderDecoder as Generator

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

class Inference():
    def __init__(self):
        super(Inference, self).__init__()
        self.generator_path = 'jyjeon/weights/generator/1215_1038exponential095_weightdecay0_SSIM1_LPIPS_Focal_stridedE_valsplit/last_weights.pth.tar'
        self.segmentor_path = 'jyjeon/weights/fusion/1220_1640_base/best_miou.pth.tar'

        self.set_cuda()
        self.set_loader()

        cfg_path = 'data_loader/semantic-kitti.yaml'
        CFG = yaml.safe_load(open(cfg_path, 'r'))
        color_dict = CFG['color_map']
        learning_map = CFG['learning_map']
        learning_map_inv = CFG['learning_map_inv']
        self.color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

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
        s_info = torch.load(self.segmentor_path)
        self.segmentor.load_state_dict(s_info['model_state_dict'], strict=True)
        for param in self.segmentor.parameters():
            param.requires_grad = False

        self.epoch = s_info['epoch']
        self.mode = self.segmentor_path.split('/')[-1].split('.')[0]
        if self.mode == 'best_miou':
            self.miou = str(s_info['best_psnr'].item())[2:6]
        elif self.mode == 'last_weights':
            self.miou = str(s_info['last_miou'])[2:6]
        elif self.mode == 'earlystop':
            self.miou = 'none'

        self.criterion = FocalLosswithLovaszRegularizer(ignore_index=0)

    def set_init(self):
        self.set_model()
        self.set_dir()
        
        self.loss1_run = AverageMeter("Loss1", ":.4f")
        self.loss2_run = AverageMeter("Loss2", ":.4f")
        self.miou1_run = AverageMeter("mIoU1", ":.4f")
        self.miou2_run = AverageMeter("mIoU2", ":.4f")
        self.car1_run = AverageMeter('car', ':.4f')
        self.bicycle1_run = AverageMeter('bicycle', ':.4f')
        self.motorcycle1_run = AverageMeter('motorcycle', ':.4f')
        self.truck1_run = AverageMeter('trunk', ':.4f')
        self.other_vehicle1_run = AverageMeter('other_vehicle', ':.4f')
        self.person1_run = AverageMeter('person', ':.4f')
        self.bicyclist1_run = AverageMeter('bicyclist', ':.4f')
        self.motorcyclist1_run = AverageMeter('motorcyclist', ':.4f')
        self.road1_run = AverageMeter('road', ':.4f')
        self.parking1_run = AverageMeter('parking', ':.4f')
        self.sidewalk1_run = AverageMeter('sidewalk', ':.4f')
        self.other_ground1_run = AverageMeter('other_ground', ':.4f')
        self.building1_run = AverageMeter('building', ':.4f')
        self.fence1_run = AverageMeter('fence', ':.4f')
        self.vegetation1_run = AverageMeter('vegetation', ':.4f')
        self.trunk1_run = AverageMeter('trunk', ':.4f')
        self.terrain1_run = AverageMeter('terrain', ':.4f')
        self.pole1_run = AverageMeter('pole', ':.4f')
        self.traffic_sign1_run = AverageMeter('traiffic_sign', ':.4f')

        self.car2_run = AverageMeter('car', ':.4f')
        self.bicycle2_run = AverageMeter('bicycle', ':.4f')
        self.motorcycle2_run = AverageMeter('motorcycle', ':.4f')
        self.truck2_run = AverageMeter('trunk', ':.4f')
        self.other_vehicle2_run = AverageMeter('other_vehicle', ':.4f')
        self.person2_run = AverageMeter('person', ':.4f')
        self.bicyclist2_run = AverageMeter('bicyclist', ':.4f')
        self.motorcyclist2_run = AverageMeter('motorcyclist', ':.4f')
        self.road2_run = AverageMeter('road', ':.4f')
        self.parking2_run = AverageMeter('parking', ':.4f')
        self.sidewalk2_run = AverageMeter('sidewalk', ':.4f')
        self.other_ground2_run = AverageMeter('other_ground', ':.4f')
        self.building2_run = AverageMeter('building', ':.4f')
        self.fence2_run = AverageMeter('fence', ':.4f')
        self.vegetation2_run = AverageMeter('vegetation', ':.4f')
        self.trunk2_run = AverageMeter('trunk', ':.4f')
        self.terrain2_run = AverageMeter('terrain', ':.4f')
        self.pole2_run = AverageMeter('pole', ':.4f')
        self.traffic_sign2_run = AverageMeter('traiffic_sign', ':.4f')

        self.loss1_run.reset()
        self.loss2_run.reset()
        self.miou1_run.reset()
        self.miou2_run.reset()
        self.car1_run.reset()
        self.bicycle1_run.reset()
        self.motorcycle1_run.reset()
        self.truck1_run.reset()
        self.other_vehicle1_run.reset()
        self.person1_run.reset()
        self.bicyclist1_run.reset()
        self.motorcyclist1_run.reset()
        self.road1_run.reset()
        self.parking1_run.reset()
        self.sidewalk1_run.reset()
        self.other_ground1_run.reset()
        self.building1_run.reset()
        self.fence1_run.reset()
        self.vegetation1_run.reset()
        self.trunk1_run.reset()
        self.terrain1_run.reset()
        self.pole1_run.reset()
        self.traffic_sign1_run.reset()

        self.car2_run.reset()
        self.bicycle2_run.reset()
        self.motorcycle2_run.reset()
        self.truck2_run.reset()
        self.other_vehicle2_run.reset()
        self.person2_run.reset()
        self.bicyclist2_run.reset()
        self.motorcyclist2_run.reset()
        self.road2_run.reset()
        self.parking2_run.reset()
        self.sidewalk2_run.reset()
        self.other_ground2_run.reset()
        self.building2_run.reset()
        self.fence2_run.reset()
        self.vegetation2_run.reset()
        self.trunk2_run.reset()
        self.terrain2_run.reset()
        self.pole2_run.reset()
        self.traffic_sign2_run.reset()

        self.iou1 = IOUEval(n_classes=20, ignore=0)
        self.iou2 = IOUEval(n_classes=20, ignore=0)

    def set_dir(self):
        if 'earlystop' in self.segmentor_path :
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')[:-1]), f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_testset")
        elif 'miou' in self.segmentor_path :
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')[:-1]), f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_miou{self.miou}_testset")
        elif 'last' in self.segmentor_path:
            self.save_dir = os.path.join(('/').join(self.segmentor_path.split('/')[:-1]), f"{time.strftime('%m%d_%H%M')}_{self.mode}_{self.epoch}epoch_miou{self.miou}_testset")

        if not os.path.exists(self.save_dir):
            # os.makedirs(os.path.join(self.save_dir, 'outrgb'))
            # os.makedirs(os.path.join(self.save_dir, 'outsegment'))
            # os.makedirs(os.path.join(self.save_dir, 'inputrdm'))
            # os.makedirs(os.path.join(self.save_dir, 'label'))
            os.makedirs(os.path.join(self.save_dir, '0'))
            os.makedirs(os.path.join(self.save_dir, '1'))
            os.makedirs(os.path.join(self.save_dir, '2'))
            os.makedirs(os.path.join(self.save_dir, '3'))
            os.makedirs(os.path.join(self.save_dir, '4'))

    def set_loader(self):
        self.data_path = 'data/semantic_kitti/kitti/dataset/sequences'

        dataset = SemanticKITTI(self.data_path, shape=(384, 1248), nclasses=20, 
                                mode='valid', front=360, split=False, crop_size=128, thetas=[0, 81, 161, 241, 321])
        self.loader = DataLoader(dataset, batch_size=20, num_workers=self.num_workers, shuffle=False)

    def convert_color(self, arr, color_dict):
        result = np.zeros((*arr.shape, 3))
        for b in range(result.shape[0]):
            for i in color_dict:
                j = np.where(arr[b]==i)
                try:
                    xs, ys = j[0], j[1]
                except:
                    xs = j[0]

                if len(xs) == 0:
                    continue
                for x, y in zip(xs, ys):
                    result[b,x,y,2] = color_dict[i][0]
                    result[b,x,y,1] = color_dict[i][1]
                    result[b,x,y,0] = color_dict[i][2]

        return result

    def logs(self):
        loss1, loss2, miou1, miou2 = self.loss1_run.avg, self.loss2_run.avg, self.miou1_run.avg, self.miou2_run.avg
        car1, bicycle1, motorcycle1, truck1, other_vehicle1, = self.car1_run.avg, self.bicycle1_run.avg, self.motorcycle1_run.avg, self.truck1_run.avg, self.other_vehicle1_run.avg
        person1, bicyclist1, motorcyclist1, road1, parking1 = self.person1_run.avg, self.bicyclist1_run.avg, self.motorcyclist1_run.avg, self.road1_run.avg, self.parking1_run.avg
        sidewalk1, other_ground1, building1, fence1, vegetation1 = self.sidewalk1_run.avg, self.other_ground1_run.avg, self.building1_run.avg, self.fence1_run.avg, self.vegetation1_run.avg,
        trunk1, terrain1, pole1, traffic_sign1 = self.trunk1_run.avg, self.terrain1_run.avg, self.pole1_run.avg, self.traffic_sign1_run.avg

        car2, bicycle2, motorcycle2, truck2, other_vehicle2, = self.car2_run.avg, self.bicycle2_run.avg, self.motorcycle2_run.avg, self.truck2_run.avg, self.other_vehicle2_run.avg
        person2, bicyclist2, motorcyclist2, road2, parking2 = self.person2_run.avg, self.bicyclist2_run.avg, self.motorcyclist2_run.avg, self.road2_run.avg, self.parking2_run.avg
        sidewalk2, other_ground2, building2, fence2, vegetation2 = self.sidewalk2_run.avg, self.other_ground2_run.avg, self.building2_run.avg, self.fence2_run.avg, self.vegetation2_run.avg,
        trunk2, terrain2, pole2, traffic_sign2 = self.trunk2_run.avg, self.terrain2_run.avg, self.pole2_run.avg, self.traffic_sign2_run.avg


        print('\n[Test] | Loss1 {:.4f} | Loss2 {:.4f} | mIoU1 {:.4f} | mIoU2 {:.4f} '.format(loss1, loss2, miou1, miou2))

        print(f'\nmIoU1\t\t{miou1}\t\tLoss1\t{loss1}')
        print(f'\nmIoU2\t\t{miou2}\t\tLoss2\t{loss2}')
        print('\ncar1\t\t\t{:.4f}'.format(car1))
        print('car2\t\t\t{:.4f}'.format(car2))
        print('bicycle1\t\t\t{:.4f}'.format(bicycle1))
        print('bicycle2\t\t\t{:.4f}'.format(bicycle2))
        print('motorcycle1\t\t{:.4f}'.format(motorcycle1))
        print('motorcycle2\t\t{:.4f}'.format(motorcycle2))
        print('truck1\t\t\t{:.4f}'.format(truck1))
        print('truck2\t\t\t{:.4f}'.format(truck2))
        print('other_vehicle1\t\t{:.4f}'.format(other_vehicle1))
        print('other_vehicle2\t\t{:.4f}'.format(other_vehicle2))
        print('person1\t\t\t{:.4f}'.format(person1))
        print('person2\t\t\t{:.4f}'.format(person2))
        print('bicyclist1\t\t{:.4f}'.format(bicyclist1))
        print('bicyclist2\t\t{:.4f}'.format(bicyclist2))
        print('motorcyclist1\t\t{:.4f}'.format(motorcyclist1))
        print('motorcyclist2\t\t{:.4f}'.format(motorcyclist2))
        print('road1\t\t\t{:.4f}'.format(road1))
        print('road2\t\t\t{:.4f}'.format(road2))
        print('parking1\t\t{:.4f}'.format(parking1))
        print('parking2\t\t{:.4f}'.format(parking2))
        print('sidewalk1\t\t{:.4f}'.format(sidewalk1))
        print('sidewalk2\t\t{:.4f}'.format(sidewalk2))
        print('other ground1\t\t{:.4f}'.format(other_ground1))
        print('other ground2\t\t{:.4f}'.format(other_ground2))
        print('building1\t\t{:.4f}'.format(building1))
        print('building2\t\t{:.4f}'.format(building2))
        print('fence1\t\t\t{:.4f}'.format(fence1))
        print('fence2\t\t\t{:.4f}'.format(fence2))
        print('vegetation1\t\t{:.4f}'.format(vegetation1))
        print('vegetation2\t\t{:.4f}'.format(vegetation2))
        print('trunk1\t\t\t{:.4f}'.format(trunk1))
        print('trunk2\t\t\t{:.4f}'.format(trunk2))
        print('terrain1\t\t\t{:.4f}'.format(terrain1))
        print('terrain2\t\t\t{:.4f}'.format(terrain2))
        print('pole1\t\t\t{:.4f}'.format(pole1))
        print('pole2\t\t\t{:.4f}'.format(pole2))
        print('traffic_sign1\t\t{:.4f}'.format(traffic_sign1))
        print('traffic_sign2\t\t{:.4f}'.format(traffic_sign2))

        print('\nEND\n')      

        with open(f'{self.save_dir}/result.txt','w') as f:
            f.write(f'\nmIoU1\t\t{miou1}\t\tLoss1\t{loss1}')
            f.write(f'\nmIoU2\t\t{miou2}\t\tLoss2\t{loss2}')
            f.write('\nIoU per Class Result-----------------\n\n')
            f.write(f'car1\t\t\t: {car1:.4f}\n')
            f.write(f'car2\t\t\t: {car2:.4f}\n')
            f.write(f'bicycle1\t\t\t: {bicycle1:.4f}\n') 
            f.write(f'bicycle2\t\t\t: {bicycle2:.4f}\n') 
            f.write(f'motorcycle1\t\t: {motorcycle1:.4f}\n') 
            f.write(f'motorcycle2\t\t: {motorcycle2:.4f}\n') 
            f.write(f'truck1\t\t\t: {truck1:.4f}\n') 
            f.write(f'truck2\t\t\t: {truck2:.4f}\n') 
            f.write(f'otehr vehicle1\t\t: {other_vehicle1:.4f}\n') 
            f.write(f'otehr vehicle2\t\t: {other_vehicle2:.4f}\n') 
            f.write(f'person1\t\t\t: {person1:.4f}\n') 
            f.write(f'person2\t\t\t: {person2:.4f}\n') 
            f.write(f'bicyclist1\t\t: {bicyclist1:.4f}\n') 
            f.write(f'bicyclist2\t\t: {bicyclist2:.4f}\n') 
            f.write(f'motorcyclist1\t\t: {motorcyclist1:.4f}\n') 
            f.write(f'motorcyclist2\t\t: {motorcyclist2:.4f}\n') 
            f.write(f'road1\t\t\t: {road1:.4f}\n') 
            f.write(f'road2\t\t\t: {road2:.4f}\n') 
            f.write(f'parking1\t\t: {parking1:.4f}\n') 
            f.write(f'parking2\t\t: {parking2:.4f}\n') 
            f.write(f'sidewalk1\t\t: {sidewalk1:.4f}\n') 
            f.write(f'sidewalk2\t\t: {sidewalk2:.4f}\n') 
            f.write(f'other ground1\t\t: {other_ground1:.4f}\n') 
            f.write(f'other ground2\t\t: {other_ground2:.4f}\n') 
            f.write(f'building1\t\t: {building1:.4f}\n') 
            f.write(f'building2\t\t: {building2:.4f}\n') 
            f.write(f'fence1\t\t\t: {fence1:.4f}\n') 
            f.write(f'fence2\t\t\t: {fence2:.4f}\n') 
            f.write(f'vegetation1\t\t: {vegetation1:.4f}\n') 
            f.write(f'vegetation2\t\t: {vegetation2:.4f}\n') 
            f.write(f'trunk1\t\t\t: {trunk1:.4f}\n') 
            f.write(f'trunk2\t\t\t: {trunk2:.4f}\n') 
            f.write(f'terrain1\t\t\t: {terrain1:.4f}\n') 
            f.write(f'terrain2\t\t\t: {terrain2:.4f}\n') 
            f.write(f'pole1\t\t\t: {pole1:.4f}\n') 
            f.write(f'pole2\t\t\t: {pole2:.4f}\n') 
            f.write(f'traffic_sign1\t\t: {traffic_sign1:.4f}\n') 
            f.write(f'traffic_sign2\t\t: {traffic_sign2:.4f}\n') 
        f.close()
                
    def inference(self):

        self.set_init()
        progress = ProgressMeter(len(self.loader), 
                                    [self.loss1_run, self.miou1_run, self.miou2_run], 
                                    prefix=f"{self.mode} TEST ")
        
        self.generator.eval()
        self.segmentor.eval()
        self.iou1.reset()
        self.iou2.reset()
        with torch.no_grad():
            # idx = 0 
            for iter, batch in enumerate(self.loader):
                
                inputs_rdm = rearrange(batch['rdm'], 'b1 b2 c h w -> (b1 b2 ) c h w').to(self.device)
                labels_3d = rearrange(batch['label'], 'b1 b2 h w -> (b1 b2) h w').to(self.device)
                labels_c = rearrange(batch['label_c'], 'b1 b2 c h w -> (b1 b2) h w c').cpu().detach().numpy()*255
                bs = inputs_rdm.size(0)

                outs_rgb, outs_seg1 = self.generator(inputs_rdm)
                outs_seg2 = self.segmentor(outs_rgb, inputs_rdm)
                loss1 = self.criterion(outs_seg1, labels_3d).detach()
                loss2 = self.criterion(outs_seg2, labels_3d).detach()
                
                self.iou1.addBatch(torch.argmax(outs_seg1, 1), labels_3d)
                self.iou2.addBatch(torch.argmax(outs_seg2, 1), labels_3d)
                miou1, per_iou1 = self.iou1.getIoU()
                miou2, per_iou2 = self.iou2.getIoU()

                self.miou1_run.update(miou1, bs)
                self.loss1_run.update(loss1.item(), bs)
                self.car1_run.update(per_iou1[1].item(), bs)
                self.bicycle1_run.update(per_iou1[2].item(), bs)
                self.motorcycle1_run.update(per_iou1[3].item(), bs)
                self.truck1_run.update(per_iou1[4].item(), bs)
                self.other_vehicle1_run.update(per_iou1[5].item(), bs)
                self.person1_run.update(per_iou1[6].item(), bs)
                self.bicyclist1_run.update(per_iou1[7].item(), bs)
                self.motorcyclist1_run.update(per_iou1[8].item(), bs)
                self.road1_run.update(per_iou1[9].item(), bs)
                self.parking1_run.update(per_iou1[10].item(), bs)
                self.sidewalk1_run.update(per_iou1[11].item(), bs)
                self.other_ground1_run.update(per_iou1[12].item(), bs)
                self.building1_run.update(per_iou1[13].item(), bs)
                self.fence1_run.update(per_iou1[14].item(), bs)
                self.vegetation1_run.update(per_iou1[15].item(), bs)
                self.trunk1_run.update(per_iou1[16].item(), bs)
                self.terrain1_run.update(per_iou1[17].item(), bs)
                self.pole1_run.update(per_iou1[18].item(), bs)
                self.traffic_sign1_run.update(per_iou1[19].item(), bs)

                self.miou2_run.update(miou2, bs)
                self.loss2_run.update(loss2.item(), bs)
                self.car2_run.update(per_iou2[1].item(), bs)
                self.bicycle2_run.update(per_iou2[2].item(), bs)
                self.motorcycle2_run.update(per_iou2[3].item(), bs)
                self.truck2_run.update(per_iou2[4].item(), bs)
                self.other_vehicle2_run.update(per_iou2[5].item(), bs)
                self.person2_run.update(per_iou2[6].item(), bs)
                self.bicyclist2_run.update(per_iou2[7].item(), bs)
                self.motorcyclist2_run.update(per_iou2[8].item(), bs)
                self.road2_run.update(per_iou2[9].item(), bs)
                self.parking2_run.update(per_iou2[10].item(), bs)
                self.sidewalk2_run.update(per_iou2[11].item(), bs)
                self.other_ground2_run.update(per_iou2[12].item(), bs)
                self.building2_run.update(per_iou2[13].item(), bs)
                self.fence2_run.update(per_iou2[14].item(), bs)
                self.vegetation2_run.update(per_iou2[15].item(), bs)
                self.trunk2_run.update(per_iou2[16].item(), bs)
                self.terrain2_run.update(per_iou2[17].item(), bs)
                self.pole2_run.update(per_iou2[18].item(), bs)
                self.traffic_sign2_run.update(per_iou2[19].item(), bs)
                
                progress.display(iter)

                # outs_seg1 = self.convert_color(torch.argmax(outs_seg1, 1).cpu().detach().numpy(), self.color_dict) # b h w c
                # outs_seg2 = self.convert_color(torch.argmax(outs_seg2, 1).cpu().detach().numpy(), self.color_dict)
                # inputs_rdm= rearrange(inputs_rdm, 'b c h w -> b h w c').cpu().detach().numpy()
                # outs_rgb = rearrange(outs_rgb, 'b c h w -> b h w c').cpu().detach().numpy()*255
                # labels_3d = labels_3d.cpu().detach().numpy()
                    
                # for i in range(5):
                #     result = np.concatenate([inputs_rdm[i], outs_seg1[i], outs_rgb[i], outs_seg2[i], labels_c[i]], axis=-2)
                #     cv2.imwrite(f"{self.save_dir}/{str(i)}/{str(iter)}.png", result)

                # cv2.imwrite(f"{self.save_dir}/0/segment_0.png", self.convert_color(torch.argmax(outs_seg, 1)[0][0].cpu().detach().numpy(), self.color_dict))
                # cv2.imwrite(f"{self.save_dir}/sample/segment_1.png", self.convert_color(torch.argmax(outs_seg, 1)[0][1].cpu().detach().numpy(), self.color_dict))
                # cv2.imwrite(f"{self.save_dir}/sample/segment_2.png", self.convert_color(torch.argmax(outs_seg, 1)[0][2].cpu().detach().numpy(), self.color_dict))
                # cv2.imwrite(f"{self.save_dir}/sample/segment_3.png", self.convert_color(torch.argmax(outs_seg, 1)[0][3].cpu().detach().numpy(), self.color_dict))
                # cv2.imwrite(f"{self.save_dir}/sample/segment_4.png", self.convert_color(torch.argmax(outs_seg, 1)[0][4].cpu().detach().numpy(), self.color_dict))
            self.logs()


if __name__ == "__main__":
    Inference().inference()