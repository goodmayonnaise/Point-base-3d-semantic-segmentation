import cv2, yaml
import numpy as np 
from einops import rearrange
from time import time

from utils.logs import AverageMeter, ProgressMeter
from utils.metrics import psnr, IOUEval
import torch 
# import torchvision

class Training():
    def __init__(self, model, epochs, train_loader, val_loader, device, optimizer, criterion, criterion2, criterion3,
                 scheduler, save_path, earlystop, metrics, writer_train, writer_val, epoch=None, mode=None, train_mode=True):
        super(Training, self).__init__()
        self.mode = mode
        self.train_mode = train_mode

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion2 = criterion2
        self.criterion3 = criterion3
        self.save_path = save_path
        self.earlystop = earlystop
        self.device = device
        self.metrics = metrics
        self.writer_train = writer_train
        self.writer_val = writer_val
        self.scheduler = scheduler
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_lpips = 1.0
        self.best_miou = 0.0

        self.b_t = AverageMeter('time', ':6.3f')
        self.loss_run = AverageMeter('loss', ':.4f')
        self.lpips_run = AverageMeter("LPIPS", ':.4f')
        self.ssim_run = AverageMeter("SSIM", ":.4f")
        self.psnr_run = AverageMeter('PSNR', ':.4f')
        self.segment_run = AverageMeter("LSeg",':.4f')
        self.miou_run = AverageMeter("mIoU", ":.4f")

        self.miou = IOUEval(n_classes=20, ignore=0)
        self.reset()

        cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
        CFG = yaml.safe_load(open(cfg_path, 'r'))
        color_dict = CFG['color_map']
        learning_map = CFG['learning_map']
        learning_map_inv = CFG['learning_map_inv']
        self.color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

        if epoch is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch 

    def reset(self):
        self.b_t.reset()
        self.loss_run.reset()
        self.lpips_run.reset()
        self.psnr_run.reset()
        self.ssim_run.reset()
        self.segment_run.reset()
        self.miou.reset()

    def save(self, epoch, mode, **kwargs):
        if mode == 'best_psnr':
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_psnr': self.best_psnr,
                'metrics': self.metrics}, f"{self.save_path}/best_psnr.pth.tar")
        elif mode == "best_ssim":
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_ssim': self.best_ssim,
                'metrics': self.metrics}, f"{self.save_path}/best_ssim.pth.tar")
        elif mode == "best_lpips":
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_lpips': self.best_lpips,
                'metrics': self.metrics}, f"{self.save_path}/best_lpips.pth.tar")
        elif mode == 'last':
            v_lpips = kwargs['v_lpips']
            v_psnr = kwargs['v_psnr']
            v_ssim = kwargs['v_ssim']
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_psnr': self.best_psnr,
                        'best_ssim':self.best_ssim,
                        'last_v_ssim':v_ssim,
                        'last_val_psnr': v_psnr,
                        'last_v_lpips':v_lpips,
                        'metrics': self.metrics,
                        }, f'./{self.save_path}/last_weights.pth.tar')

    def convert_color(self, arr):
        arr = arr.detach().cpu().numpy()
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
                for x, y in zip(xs, ys):
                    result[b,x,y,0] = self.color_dict[i][0]
                    result[b,x,y,1] = self.color_dict[i][1]
                    result[b,x,y,2] = self.color_dict[i][2]

        return result

    def train(self, epoch, since=time()):
        
            epoch += self.start_epoch 
            print(f'\n-----------------------\nEpoch {epoch}')
            
            progress = ProgressMeter(len(self.train_loader),
                                    [self.b_t, self.loss_run, self.lpips_run, self.segment_run, self.psnr_run, self.ssim_run, self.miou_run],
                                    prefix=f"epoch {epoch} Train")

            self.model.train()
            end = time()
            self.miou.reset()
            for iter, batch in enumerate(self.train_loader):            
                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    inputs = batch['rdm'].to(self.device)
                    labels_rgb = batch['img'].to(self.device) # rgb 3channel
                    labels_seg = batch['label'].to(self.device)
                    with torch.autograd.set_detect_anomaly(True):
                        outs_rgb, outs_seg = self.model(inputs) # rgb 3channel 
                        ssim = -self.criterion(outs_rgb, labels_rgb/255) 
                        lpips = self.criterion2(outs_rgb, labels_rgb/255, normalize=True)
                        segment = self.criterion3(outs_seg, labels_seg)
                        
                        ssim.backward(retain_graph=True) # loss
                        lpips.mean().backward(retain_graph=True)
                        segment.backward()
                        self.optimizer.step()

                # statistics                        
                bs = inputs.size(0)
                self.loss_run.update(1+ssim.item(), bs)
                self.lpips_run.update(lpips.mean().item(), bs)
                self.ssim_run.update(-ssim.item(), bs)
                self.psnr_run.update(psnr(outs_rgb, labels_rgb/255), bs)
                self.segment_run.update(segment.item(), bs)

                self.miou.addBatch(torch.argmax(outs_seg, 1), labels_seg)
                miou, per_iou = self.miou.getIoU()
                self.miou_run.update(miou.item(), bs)

                self.b_t.update(time()-end, bs)
                end = time()

                progress.display(iter)

                inputs = rearrange(inputs[-1], 'c h w -> h w c').detach().cpu().numpy()
                outs_rgb = rearrange(outs_rgb[-1], 'c h w -> h w c').cpu().detach().numpy()*255
                labels_rgb = rearrange(labels_rgb[-1], 'c h w -> h w c').cpu().detach().numpy()
                outs_seg = self.convert_color(torch.argmax(outs_seg, dim=1))[-1]
                labels_seg = rearrange(batch['label_c'][-1], 'c h w -> h w c').detach().cpu().numpy()*255
                vis = np.concatenate([inputs, labels_rgb, outs_rgb, labels_seg, outs_seg], axis=0)
                cv2.imwrite(f"{self.save_path}/real_time.png", vis)

            if (epoch == 1) or (epoch % 10) == 0 :
                cv2.imwrite(f"{self.save_path}/samples/train/{epoch}.png", vis)
                
            t_loss, t_psnr, t_ssim, t_lpips, t_segment, t_miou = self.loss_run.avg,  self.psnr_run.avg, self.ssim_run.avg, self.lpips_run.avg, self.segment_run.avg, self.miou_run.avg

            self.metrics['t_loss'].append(t_loss)            
            self.metrics['t_lpips'].append(t_lpips)
            self.metrics['t_psnr'].append(t_psnr)
            self.metrics['t_ssim'].append(t_ssim)
            self.metrics['t_segment'].append(t_segment)
            self.metrics['t_miou'].append(t_miou)
            
            print('\ntrain | LSSIM {:.4f} | LPIPS {:.4f} | Lseg {:.4f} | PSNR {:.4f} | SSIM {:.4f} | mIoU {:.4f} \n'.format(t_loss,  t_lpips, t_segment,  t_psnr, t_ssim, t_miou))
            
            v_loss,  v_psnr, self.v_ssim, self.v_lpips, v_segment, self.v_miou = self.val(epoch)
            self.metrics['v_loss'].append(v_loss)
            self.metrics['v_lpips'].append(self.v_lpips)
            self.metrics['v_segment'].append(v_segment)
            self.metrics['v_psnr'].append(v_psnr)
            self.metrics['v_ssim'].append(self.v_ssim)
            self.metrics['v_miou'].append(self.v_miou)

            self.scheduler.step()
            
            # save history
            time_elapsed = time() - since
            with open(f'./{self.save_path}/result.csv', 'a') as epoch_log:
                epoch_log.write('\n{} \t\t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.0f}m {:.0f}s'\
                                .format(epoch, t_loss, v_loss, t_lpips, self.v_lpips, t_segment, v_segment, t_psnr, v_psnr, t_ssim, self.v_ssim, t_miou, self.v_miou, time_elapsed//60, time_elapsed % 60))

            # save model per epochs         --------------------------------------------------
            if self.train_mode:
                self.save(epoch, 'last', v_psnr=v_psnr, v_ssim=self.v_ssim, v_lpips=self.v_lpips)

            # Save best psnr model to file       --------------------------------------------------
            if v_psnr > self.best_psnr:
                print('PSNR improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_psnr, v_psnr, epoch))
                self.best_psnr = v_psnr
                if self.train_mode:
                    self.save(epoch, 'best_psnr', best_psnr=self.best_psnr)
            else:
                print('PSNR NOT improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_psnr, v_psnr, epoch))

            # Save best ssim model to file       --------------------------------------------------
            if self.v_ssim > self.best_ssim:
                print('SSIM improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_ssim, self.v_ssim, epoch))
                self.best_ssim = self.v_ssim
                if self.train_mode:
                    self.save(epoch, 'best_ssim', best_ssim=self.best_ssim)
            else:
                print('SSIM NOT improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_ssim, self.v_ssim, epoch))

            if self.v_lpips < self.best_lpips:
                print('LPIPS decreased from {:.4f} to {:.4f}. epoch {}'.format(self.best_lpips, self.v_lpips, epoch))
                self.best_lpips = self.v_lpips
                if self.train_mode:
                    self.save(epoch, 'best_lpips', best_lpips=self.best_lpips)
            else:
                print('LPIPS NOT decreased from {:.4f} to {:.4f}. epoch {}'.format(self.best_lpips, self.v_lpips, epoch))

            if self.v_miou > self.best_miou:
                print('mIoU improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_miou, self.v_miou, epoch))
                self.best_miou = self.v_miou
                if self.train_mode:
                    self.save(epoch, 'best_miou', best_miou=self.best_miou)
            else:
                print('mIoU NOT improved from {:.4f} to {:.4f}. epoch {}'.format(self.best_miou, self.v_miou, epoch))

            # tensorboard                   --------------------------------------------------
            if self.writer_train != None:
                self.writer_train.add_scalar("Loss", t_loss, epoch)
                self.writer_train.add_scalar("LPIPS", t_lpips, epoch)
                self.writer_train.add_scalar("LSeg", t_segment, epoch)
                self.writer_train.add_scalar("PSNR", t_psnr, epoch)
                self.writer_train.add_scalar("SSIM", t_ssim, epoch)
                self.writer_train.add_scalar("mIoU", t_miou, epoch)

                self.writer_val.add_scalar("Loss", v_loss, epoch)
                self.writer_val.add_scalar("LPIPS", self.v_lpips, epoch)
                self.writer_val.add_scalar("LSeg", v_segment, epoch)
                self.writer_val.add_scalar("PSNR", v_psnr, epoch)
                self.writer_val.add_scalar("SSIM", self.v_ssim, epoch)                
                self.writer_val.add_scalar("mIoU", self.v_miou, epoch)
                self.writer_val.add_scalar("optim", self.optimizer.param_groups[0]['lr'], epoch)      

                self.writer_train.flush()
                self.writer_train.close()
                self.writer_val.flush()
                self.writer_val.close()

            time_elapsed = time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            epoch += 1
            if self.mode == 'while':
                return epoch 


    def val(self, epoch):
        self.reset()
        self.miou.reset()

        progress = ProgressMeter(len(self.val_loader),
                                [self.b_t, self.loss_run, self.lpips_run, self.segment_run, self.psnr_run, self.ssim_run, self.miou_run],
                                prefix=f"epoch {epoch} Test")
        
        self.model.eval()
        with torch.no_grad():
            end = time()
            for iter, batch in enumerate(self.val_loader):

                inputs = batch['rdm'].to(self.device)
                labels_rgb = batch['img'].to(self.device)
                labels_seg = batch['label'].to(self.device)

                outs_rgb, outs_seg = self.model(inputs) # rgb 3channel 

                ssim = -self.criterion(outs_rgb, labels_rgb/255).detach() 
                lpips = self.criterion2(outs_rgb, labels_rgb/255, normalize=True)
                segment = self.criterion3(outs_seg, labels_seg)
                bs = inputs.size(0)
                
                self.loss_run.update(1+ssim.item(), bs)
                self.lpips_run.update(lpips.mean().item(), bs)
                self.segment_run.update(segment.item(), bs)
                self.ssim_run.update(-ssim.item(), bs)
                self.psnr_run.update(psnr(outs_rgb, labels_rgb/255), bs)

                self.miou.addBatch(torch.argmax(outs_seg, 1), labels_seg)
                miou, per_iou = self.miou.getIoU()
                self.miou_run.update(miou.item(), bs)

                self.b_t.update(time()-end, bs)
                end = time()

                progress.display(iter)

                inputs = rearrange(inputs[-1], 'c h w -> h w c').detach().cpu().numpy()
                outs_rgb = rearrange(outs_rgb[-1], 'c h w -> h w c').cpu().detach().numpy()*255
                labels_rgb = rearrange(labels_rgb[-1], 'c h w -> h w c').cpu().detach().numpy()
                outs_seg = self.convert_color(torch.argmax(outs_seg, dim=1))[-1]
                labels_seg = rearrange(batch['label_c'][-1], 'c h w -> h w c').detach().cpu().numpy()*255
                vis = np.concatenate([inputs, labels_rgb, outs_rgb, labels_seg, outs_seg], axis=0)
                cv2.imwrite(f"{self.save_path}/real_time.png", vis)

        if (epoch == 1) or (epoch % 10) == 0 :
            cv2.imwrite(f"{self.save_path}/samples/val/{epoch}.png", vis)

        v_loss, v_psnr, v_ssim, v_lpips, v_segment, v_miou = self.loss_run.avg, self.psnr_run.avg, self.ssim_run.avg,  self.lpips_run.avg, self.segment_run.avg, self.miou_run.avg
            
        print('\nvalidation | LSSIM {:.4f} | LPIPS {:.4f} | Lseg {:.4f} | PSNR {:.4f} | SSIM {:.4f} | mIoU {:.4f}\n'.format(v_loss, v_lpips, v_segment, v_psnr, v_ssim, v_miou))
        return v_loss, v_psnr, v_ssim, v_lpips, v_segment, v_miou

    def start_train(self):
        if self.mode =='while':
            self.v_ssim = 0.0
            epoch = 1
            while self.v_ssim < 0.95:
                epoch = self.train(epoch)   
        elif self.mode == "epoch":
            for epoch in range(1, self.epochs+1):
                self.train(epoch=epoch)         