import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import RED_CNN
from measure import compute_measure


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode

        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig


        self.REDCNN = RED_CNN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss() 
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
    # Select a single example from the batch for visualization
    # Assuming x, y, and pred are torch tensors of shape [batch_size, channels, height, width]
        x_single = x[0, 0].cpu().numpy()  # Shape: [height, width]
        y_single = y[0, 0].cpu().numpy()  # Shape: [height, width]
        pred_single = pred[0, 0].cpu().numpy()  # Shape: [height, width]

        # Visualization
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x_single, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Input Image', fontsize=30)
        ax[1].imshow(pred_single, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Predicted Image', fontsize=30)
        ax[2].imshow(y_single, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Ground Truth Image', fontsize=30)

        # Optionally, save the figure
        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close(f)  # Close the figure to free memory

    def train(self):
        # print("train called")
        train_losses = defaultdict(list)
        total_iters = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            for iter_, sample in enumerate(self.data_loader):
                total_iters += 1
                x = sample['LQ'].to(self.device)  # Input images
                y = sample['HQ'].to(self.device)  # Ground truth images

                # Forward pass
                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses[epoch].append(loss.item())

                # Logging
                if total_iters % self.print_iters == 0:
                    print(f"Epoch [{epoch}/{self.num_epochs}], Step [{iter_ + 1}/{len(self.data_loader)}], Loss: {loss.item():.4f}, Time: {time.time() - start_time:.1f}s", flush=True)

                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # Save model periodically
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
        print("~~~~~~~Training completed~~~~~~~~~")
        # Optionally save training losses for analysis
        np.save(os.path.join(self.save_path, 'train_losses.npy'), np.array(train_losses))


    def test(self):
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, batch_samples in enumerate(self.data_loader):
                x = batch_samples['LQ'].to(self.device) # Input images
                y = batch_samples['HQ'].to(self.device) # Ground truth images
                # x = x.unsqueeze(0).float().to(self.device)
                # y = y.unsqueeze(0).float().to(self.device)

                # Assuming x, y, and pred are batch tensors with shapes [batch_size, channels, height, width]
                # x = x[0].squeeze().cpu().detach().numpy()
                # y = y[0].squeeze().cpu().detach().numpy()
                # pred = self.REDCNN(x)
                # pred = pred[0].squeeze().cpu().detach().numpy()

                # x = x.unsqueeze(0).float().to(self.device)
                # y = y.unsqueeze(0).float().to(self.device)
                pred = self.REDCNN(x)
                # pred = pred.unsqueeze(0).float().to(self.device)

                # Assume shape_ is for reshaping or viewing the tensor. Adjust as needed.
                shape_ = x.shape[-1] 

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.cpu().detach()))
                y = self.trunc(self.denormalize_(y.cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.cpu().detach()))

                # x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                # y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                prefix="Compute measurements ..",
                                suffix='Complete', length=25)
        print('\n')
        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                pred_ssim_avg/len(self.data_loader), 
                                                                                                pred_rmse_avg/len(self.data_loader)))

    # def test(self):
    #     del self.REDCNN
    #     # load
    #     self.REDCNN = RED_CNN().to(self.device)
    #     self.load_model(self.test_iters)

    #     # compute PSNR, SSIM, RMSE
    #     ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    #     pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(self.data_loader):
    #             shape_ = x.shape[-1]
    #             #Commented unsqueeze
    #             # x = x.unsqueeze(0).float().to(self.device)
    #             # y = y.unsqueeze(0).float().to(self.device)

    #             pred = self.REDCNN(x)

    #             # denormalize, truncate
    #             x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
    #             y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
    #             pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

    #             data_range = self.trunc_max - self.trunc_min

    #             original_result, pred_result = compute_measure(x, y, pred, data_range)
    #             ori_psnr_avg += original_result[0]
    #             ori_ssim_avg += original_result[1]
    #             ori_rmse_avg += original_result[2]
    #             pred_psnr_avg += pred_result[0]
    #             pred_ssim_avg += pred_result[1]
    #             pred_rmse_avg += pred_result[2]

    #             # save result figure
    #             if self.result_fig:
    #                 self.save_fig(x, y, pred, i, original_result, pred_result)

    #             printProgressBar(i, len(self.data_loader),
    #                              prefix="Compute measurements ..",
    #                              suffix='Complete', length=25)
    #         print('\n')
    #         print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
    #                                                                                         ori_ssim_avg/len(self.data_loader), 
    #                                                                                         ori_rmse_avg/len(self.data_loader)))
    #         print('\n')
    #         print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
    #                                                                                               pred_ssim_avg/len(self.data_loader), 
    #                                                                                               pred_rmse_avg/len(self.data_loader)))
