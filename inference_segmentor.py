import os, cv2, yaml, time
import numpy as np 
from datetime import timedelta
from models.segmentor import EncoderDecoder
from dataloader.semantic_kitti_segmentor import SemanticKITTI
from utils.logs import AverageMeter, ProgressMeter
from utils.metrics import IOUEval

import torch 
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

def convert_color(arr, color_dict):
    result = np.zeros((*arr.shape, 3))
    for i in color_dict:
        j = np.where(arr==i)

        try:
            xs, ys = j[0], j[1]
        except:
            xs = j[0]

        if len(xs) == 0:
            continue
        for x, y in zip(xs, ys):
            result[x,y,2] = color_dict[i][0]
            result[x,y,1] = color_dict[i][1]
            result[x,y,0] = color_dict[i][2]

    return result

def inference(model, test_loader, save_path, device):
    cfg_path = 'dataloader/semantic-kitti.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    color_dict = CFG['color_map']
    learning_map = CFG['learning_map']
    learning_map_inv = CFG['learning_map_inv']
    color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

    miou_run = AverageMeter('mIoU', ':.4f')
    car_run = AverageMeter('car', ':.4f')
    bicycle_run = AverageMeter('bicycle', ':.4f')
    motorcycle_run = AverageMeter('motorcycle', ':.4f')
    truck_run = AverageMeter('trunk', ':.4f')
    other_vehicle_run = AverageMeter('other_vehicle', ':.4f')
    person_run = AverageMeter('person', ':.4f')
    bicyclist_run = AverageMeter('bicyclist', ':.4f')
    motorcyclist_run = AverageMeter('motorcyclist', ':.4f')
    road_run = AverageMeter('road', ':.4f')
    parking_run = AverageMeter('parking', ':.4f')
    sidewalk_run = AverageMeter('sidewalk', ':.4f')
    other_ground_run = AverageMeter('other_ground', ':.4f')
    building_run = AverageMeter('building', ':.4f')
    fence_run = AverageMeter('fence', ':.4f')
    vegetation_run = AverageMeter('vegetation', ':.4f')
    trunk_run = AverageMeter('trunk', ':.4f')
    terrain_run = AverageMeter('terrain', ':.4f')
    pole_run = AverageMeter('pole', ':.4f')
    traffic_sign_run = AverageMeter('traiffic_sign', ':.4f')
    progress = ProgressMeter(len(test_loader), [miou_run])
    iou = IOUEval(20, device, ignore=0)
    
    iou.reset()
    model.eval()

    with torch.no_grad():
        for iter, batch in enumerate(test_loader):

            input_rdm = batch['rdm'].to(device)
            input_img = batch['img'].to(device)
            label = batch['3d_label'].to(device) 

            output = model(input_img, input_rdm)
            bs = input_rdm.shape[0]

            iou.addBatch(torch.argmax(output,1), label)
            miou, per_iou = iou.getIoU()
            miou_run.update(miou, bs)
            car_run.update(per_iou[1].item(), bs)
            bicycle_run.update(per_iou[2].item(), bs)
            motorcycle_run.update(per_iou[3].item(), bs)
            truck_run.update(per_iou[4].item(), bs)
            other_vehicle_run.update(per_iou[5].item(), bs)
            person_run.update(per_iou[6].item(), bs)
            bicyclist_run.update(per_iou[7].item(), bs)
            motorcyclist_run.update(per_iou[8].item(), bs)
            road_run.update(per_iou[9].item(), bs)
            parking_run.update(per_iou[10].item(), bs)
            sidewalk_run.update(per_iou[11].item(), bs)
            other_ground_run.update(per_iou[12].item(), bs)
            building_run.update(per_iou[13].item(), bs)
            fence_run.update(per_iou[14].item(), bs)
            vegetation_run.update(per_iou[15].item(), bs)
            trunk_run.update(per_iou[16].item(), bs)
            terrain_run.update(per_iou[17].item(), bs)
            pole_run.update(per_iou[18].item(), bs)
            traffic_sign_run.update(per_iou[19].item(), bs)

            cv2.imwrite(f"{save_path}/sample/segment.png", convert_color(torch.argmax(output, 1)[0].cpu().detach().numpy(), color_dict))

        
            progress.display(iter)


        miou = miou_run.avg 
        car, bicycle, motorcycle, truck, other_vehicle, = car_run.avg, bicycle_run.avg, motorcycle_run.avg, truck_run.avg, other_vehicle_run.avg
        person, bicyclist, motorcyclist, road, parking = person_run.avg, bicyclist_run.avg, motorcyclist_run.avg, road_run.avg, parking_run.avg
        sidewalk, other_ground, building, fence, vegetation = sidewalk_run.avg, other_ground_run.avg, building_run.avg, fence_run.avg, vegetation_run.avg,
        trunk, terrain, pole, traffic_sign = trunk_run.avg, terrain_run.avg, pole_run.avg, traffic_sign_run.avg

        print(f'\nmIoU\t\t\t{miou} ---------------------------------')
        print('\ncar\t\t\t\t{:.4f}'.format(car))
        print('bicycle\t\t\t\t{:.4f}'.format(bicycle))
        print('motorcycle\t\t\t{:.4f}'.format(motorcycle))
        print('truck\t\t\t\t{:.4f}'.format(truck))
        print('other_vehicle\t\t\t{:.4f}'.format(other_vehicle))
        print('person\t\t\t\t{:.4f}'.format(person))
        print('bicyclist\t\t\t{:.4f}'.format(bicyclist))
        print('motorcyclist\t\t\t{:.4f}'.format(motorcyclist))
        print('road\t\t\t\t{:.4f}'.format(road))
        print('parking\t\t\t\t{:.4f}'.format(parking))
        print('sidewalk\t\t\t{:.4f}'.format(sidewalk))
        print('other ground\t\t\t{:.4f}'.format(other_ground))
        print('building\t\t\t{:.4f}'.format(building))
        print('fence\t\t\t\t{:.4f}'.format(fence))
        print('vegetation\t\t\t{:.4f}'.format(vegetation))
        print('trunk\t\t\t\t{:.4f}'.format(trunk))
        print('terrain\t\t\t\t{:.4f}'.format(terrain))
        print('pole\t\t\t\t{:.4f}'.format(pole))
        print('traffic_sign\t\t\t{:.4f}'.format(traffic_sign))

        print('\nEND\n')      

        with open(f'{save_path}/result.txt','w') as f:
            f.write(f'\ntotal mIoU\t\t\t: {miou:.4f}\n\n')
            f.write('\nIoU per Class Result-----------------\n\n')
            f.write(f'car\t\t\t\t: {car:.4f}\n')
            f.write(f'bicycle\t\t\t: {bicycle:.4f}\n') 
            f.write(f'motorcycle\t\t: {motorcycle:.4f}\n') 
            f.write(f'truck\t\t\t: {truck:.4f}\n') 
            f.write(f'otehr vehicle\t: {other_vehicle:.4f}\n') 
            f.write(f'person\t\t\t: {person:.4f}\n') 
            f.write(f'bicyclist\t\t: {bicyclist:.4f}\n') 
            f.write(f'motorcyclist\t: {motorcyclist:.4f}\n') 
            f.write(f'road\t\t\t: {road:.4f}\n') 
            f.write(f'parking\t\t\t: {parking:.4f}\n') 
            f.write(f'sidewalk\t\t: {sidewalk:.4f}\n') 
            f.write(f'other ground\t: {other_ground:.4f}\n') 
            f.write(f'building\t\t: {building:.4f}\n') 
            f.write(f'fence\t\t\t: {fence:.4f}\n') 
            f.write(f'vegetation\t\t: {vegetation:.4f}\n') 
            f.write(f'trunk\t\t\t: {trunk:.4f}\n') 
            f.write(f'terrain\t\t\t: {terrain:.4f}\n') 
            f.write(f'pole\t\t\t: {pole:.4f}\n') 
            f.write(f'traffic_sign\t: {traffic_sign:.4f}\n') 
        f.close()

def read_model(model, ckpt, freeze, optimizer=None, resume=None, scheduler=None):

    model_info = torch.load(ckpt)
    model.load_state_dict(model_info['model_state_dict'], strict=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    epoch = model_info['epoch']

    return model, optimizer, epoch 

def main():
    
    name = "earlystop_"
    ckpt_path = 'weights/segmentor/kitti_batch6_epoch2000_FocalLosswithLovaszRegularizer_Adam/adapter_removevit_dim256_light1_1027_1039/earlystop_89_5084.pt'
    
    torch.cuda.manual_seed_all(777)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 2
    timeout = timedelta(seconds=864000)
    dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    batch_size = 1
    nclasses = 20 
    img_shape = (384, 1248)
    crop_size = 128
    input_shape = (256, 1248)  

    model = EncoderDecoder(20, input_shape, embed_dim=256)
    model = DataParallel(model.to(device), device_ids=num_gpu)
    model, _, epoch = read_model(model, ckpt_path, True)
    path = 'data/semantic_kitti/kitti/dataset/sequences'
    testset = SemanticKITTI(path, img_shape, nclasses, mode='test', front=True, split=False, crop_size=crop_size)
    test_loader = DataLoader(testset, num_workers=num_workers, batch_size=batch_size)
    
    save_path = os.path.join(('/').join(ckpt_path.split('/')[:-1]), f"{name}_{epoch}epoch_{time.strftime('%m%d_%H%M')}")
    if not os.path.exists(save_path):
         os.makedirs(os.path.join(save_path, 'samples'))
    inference(model, test_loader, save_path, device)

if __name__ == "__main__":

    main()