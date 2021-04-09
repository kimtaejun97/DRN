import torch
import numpy as np


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

from torchvision import transforms 
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import h5py
import os
import cv2
import numpy as np
from torchvision.utils import save_image
from getGazeLoss import *


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8
        self.endPoint_flag = False


    def train(self):
        label_txt = open("dataset/integrated_label.txt" , "r")
        labels = label_txt.readlines()
        batch_gaze_loss=[]
        
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
       
        self.ckp.set_epoch(epoch)

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, image_names) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()


            # forward
            sr = self.model(lr[0])
           
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

                
            # compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
            # loss_primary *=0.1
        
            
            # # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            #-----------------clear dir---------------------------
            clearDir()
            # -------------------save SR image ------------
            image_root_path = "./train_SRImage"
            
            face_path = os.path.join(image_root_path, "face")
            os.makedirs(face_path, exist_ok = True)
            for i in range(len(sr[-1])):
                save_image(sr[-1][i]/255, (os.path.join(face_path, image_names[i]+'.png')))
            
            #---------------GenerateEyePatches-------------------
            generateEyePatches()

            # ------------------compute gaze_loss----------------
            gaze_loss = computeGazeLoss(labels)
            gaze_loss = gaze_loss.detach()
            gaze_loss *=100
            print(gaze_loss)
            self.ckp.write_log("GE_Loss : "+str(gaze_loss.item()))
            batch_gaze_loss.append(gaze_loss.item())
            
            # compute total loss
            loss = gaze_loss +loss_primary +loss_dual * self.opt.dual_weight


            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        
        #------------draw gaze loss graph -----------
        epoch_gaze_loss = 0 
        for loss in batch_gaze_loss:
            epoch_gaze_loss +=loss
        epoch_gaze_loss /=self.opt.test_every
        log_path = "experiments/gaze_loss.log"
        
        if epoch ==1:
            lf =open(log_path, "w+")
        else:
            lf = open(log_path, "a")

        lf.write(str(epoch_gaze_loss)+"\n")
        lf.close()

        gaze_logs = []
        lf = open(log_path, "r")
        gaze_log = lf.readline()
        while(gaze_log !=""):
            gaze_logs.append(float(gaze_log))
            gaze_log = lf.readline()
        
        min = gaze_logs[0]
        max = gaze_logs[0]
        for i in range(len(gaze_logs)):
            if gaze_logs[i] <min:
                min = gaze_logs[i]
            if gaze_logs[i] > max:
                max = gaze_logs[i]
        
        if epoch_gaze_loss <= min and epoch_gaze_loss >=max:
            self.endPoint_flag = True

        axis = np.linspace(1, epoch, epoch)
        y = gaze_logs
        plt.xlabel('epoch')
        plt.ylabel('GE_loss')
        plt.grid(True)
        plt.plot(axis,gaze_logs, 'b-')
        plt.savefig('experiments/gaze_loss.pdf')

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                f= open('5060_flip o.txt', 'w')
                eval_psnr = 0
                eval_simm =0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:

                        psnr = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                   
                        hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                        sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                        simm = utility.SSIM(hr_numpy, sr_numpy)
                        eval_simm += simm

                        eval_psnr +=psnr

                    # save test results // SR result !
                    
                    if self.endPoint_flag:
                        if self.opt.save_results:
                            self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                eval_simm = eval_simm / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                print('SIMM:',eval_simm)

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
        