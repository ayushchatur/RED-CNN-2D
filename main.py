import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import re
import torch

# print("read correct image")
#Ayush's loader
def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org
# print("CT Dataset")
# class VolCTDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         # self.vol_list = os.listdir(root_dir)
#         #added this to not look at hidden .vscode dir
#         self.vol_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
#         self.vol_list.sort()
#         self.transform = transform

#     def __len__(self):
#         return len(self.vol_list)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         idx_mod = int(idx)
#         idx_rem = idx%4

#         HQ_vols = os.listdir(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ")
#         HQ_vols.sort()
#         LQ_vols = os.listdir(self.root_dir + "/" + self.vol_list[idx_mod] + "/LQ")
#         LQ_vols.sort()

#         input_vol = None
#         target_vol = None
#         size_img = 512
#         out_size_volume = 32
#         out_size_image = 512
#         pad = 16
#         stride_vol = (int)(len(HQ_vols)/out_size_volume)
#         rem_vol = len(HQ_vols)%out_size_volume
#         req_vol = len(HQ_vols) - rem_vol
#         # print(req_vol)
#         #print("Stride volume: ", stride_vol)
#         #print("Remainder volume: ", rem_vol)
#         if(req_vol < 60):
#             print("Required volume: ", req_vol)
#             print(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ")
#         input_file = None
#         rmin = 0
#         rmax = 1

#         cmax_in = -99999
#         cmin_in = 99999
#         cmax_target = -99999
#         cmin_target = 99999
#         input_file = self.vol_list[idx_mod]

#         #if(idx_rem==0):
#         #    input_file = "tile1_" + input_file
#         #elif(idx_rem==1):
#         #    input_file = "tile2_" + input_file
#         #elif(idx_rem==2):
#         #    input_file = "tile3_" + input_file
#         #else:
#         #    input_file = "tile4_" + input_file

#         for i in range(0, req_vol):
#             if(i%stride_vol == 0):
#                 image_target = io.imread(self.root_dir + "/" + self.vol_list[idx_mod] + "/HQ/" + HQ_vols[i])
#                 image_input = io.imread(self.root_dir + "/" + self.vol_list[idx_mod] + "/LQ/" + LQ_vols[i])
#                 image_target = image_target.astype(float)
#                 image_input = image_input.astype(float)
 
#                 if("BIMCV" in input_file):
#                     image_target = np.rot90(image_target)
#                     image_input = np.rot90(image_input)
    
#                 #image_target = torch.from_numpy(image_target.reshape((1, 1, size_img, size_img)).copy())
#                 #image_target = F.interpolate(image_target.type(torch.FloatTensor), size=(out_size_image, out_size_image))
#                 #image_input = torch.from_numpy(image_input.reshape((1, 1, size_img, size_img)).copy())
#                 #image_input = F.interpolate(image_input.type(torch.FloatTensor), size=(out_size_image, out_size_image))
    
#                 image_target = torch.from_numpy(image_target.reshape((1, 1, size_img, size_img)).copy())
#                 image_input = torch.from_numpy(image_input.reshape((1, 1, size_img, size_img)).copy())

#                 if(cmax_in < torch.max(image_input).item()):
#                     cmax_in = torch.max(image_input)
#                 if(cmin_in > torch.min(image_input).item()):
#                     cmin_in = torch.min(image_input)
#                 if(cmax_target < torch.max(image_target).item()):
#                     cmax_target = torch.max(image_target)
#                 if(cmin_target > torch.min(image_target).item()):
#                     cmin_target = torch.min(image_target)

#                 #if(idx_rem==0):
#                 #    image_target = image_target[:, :, 0:out_size_image+pad, 0:out_size_image+pad].type(torch.FloatTensor)
#                 #    image_input = image_input[:, :, 0:out_size_image+pad, 0:out_size_image+pad].type(torch.FloatTensor)
#                 #elif(idx_rem==1):
#                 #    image_target = image_target[:, :, 0:out_size_image+pad, out_size_image-pad:].type(torch.FloatTensor)
#                 #    image_input = image_input[:, :, 0:out_size_image+pad, out_size_image-pad:].type(torch.FloatTensor)
#                 #elif(idx_rem==2):
#                 #    image_target = image_target[:, :, out_size_image-pad:, 0:out_size_image+pad].type(torch.FloatTensor)
#                 #    image_input = image_input[:, :, out_size_image-pad:, 0:out_size_image+pad].type(torch.FloatTensor)
#                 #else:
#                 #    image_target = image_target[:, :, out_size_image-pad:, out_size_image-pad:].type(torch.FloatTensor)
#                 #    image_input = image_input[:, :, out_size_image-pad:, out_size_image-pad:].type(torch.FloatTensor)
                
#                 image_target = image_target.type(torch.FloatTensor)
#                 image_input = image_input.type(torch.FloatTensor)
#                 #image_target = F.interpolate(image_target.type(torch.FloatTensor), size=(out_size_image, out_size_image))
#                 #image_input = image_input[:, :, out_size_image:, out_size_image:].type(torch.FloatTensor)
#                 #image_input = F.interpolate(image_input.type(torch.FloatTensor), size=(out_size_image, out_size_image))

#                 #if self.transform:
#                 #    image_target = self.transform(image_target)
#                 #    image_input = self.transform(image_target)

#                 if(i == 0):
#                     input_vol = image_input
#                     target_vol = image_target
#                 else:
#                     input_vol = torch.cat((input_vol, image_input), dim=1)
#                     target_vol = torch.cat((target_vol, image_target), dim=1)
#                 #if(i == len(HQ_vols)-1):
#         #cmax = torch.max(input_vol).item()
#         #cmin = torch.min(input_vol).item()
#         cmax = cmax_in
#         cmin = cmin_in
#         input_vol = rmin + ((input_vol - cmin)/(cmax - cmin)*(rmax - rmin))
#         #cmax = torch.max(target_vol).item()
#         #cmin = torch.min(target_vol).item()
#         cmax = cmax_target
#         cmin = cmin_target
#         target_vol = rmin + ((target_vol - cmin)/(cmax - cmin)*(rmax - rmin))    

#         sample = {'vol': input_file,
#                   'HQ': target_vol,
#                   'LQ': input_vol}
#         return sample
class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l,  length, root_hq_vgg3 = None, root_hq_vgg1= None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        # self.data_root_h_vgg_3 = root_hq_vgg3 + "/"
        # self.data_root_h_vgg_1 = root_hq_vgg1 + "/"

        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        # self.vgg_hq_img3 = os.listdir(self.data_root_h_vgg_3)
        # self.vgg_hq_img1 = os.listdir(self.data_root_h_vgg_1)

        self.img_list_l.sort()
        self.img_list_h.sort()
        # self.vgg_hq_img3.sort()
        # self.vgg_hq_img1.sort()

        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        # self.vgg_hq_img_list3 = self.vgg_hq_img3[0:length]
        # self.vgg_hq_img_list1 = self.vgg_hq_img1[0:length]
        self.sample = dict()

    def __len__(self):
        return len(self.img_list_l)

    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1

        # print("HQ", self.data_root_h + self.img_list_h[idx])
        # print("LQ", self.data_root_l + self.img_list_l[idx])
        # image_target = read_correct_image("/groups/synergy_lab/garvit217/enhancement_data/train/LQ//BIMCV_139_image_65.tif")
        # print("test")
        # exit()
        image_target = read_correct_image(self.data_root_h + self.img_list_h[idx])
        # print("low quality {} ".format(self.data_root_h + self.img_list_h[idx]))
        # print("high quality {}".format(self.data_root_h + self.img_list_l[idx]))
        # print("hq vgg b3 {}".format(self.data_root_h_vgg + self.vgg_hq_img_list[idx]))
        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])
        # vgg_hq_img3 = np.load(self.data_root_h_vgg_3 + self.vgg_hq_img_list3[idx]) ## shape : 1,256,56,56
        # vgg_hq_img1 = np.load(self.data_root_h_vgg_1 + self.vgg_hq_img_list1[idx]) ## shape : 1,64,244,244

        input_file = self.img_list_l[idx]  ## low quality image
        assert (image_input.shape[0] == 512 and image_input.shape[1] == 512)
        assert (image_target.shape[0] == 512 and image_target.shape[1] == 512)
        cmax1 = np.amax(image_target)
        cmin1 = np.amin(image_target)
        image_target = rmin + ((image_target - cmin1) / (cmax1 - cmin1) * (rmax - rmin))
        assert ((np.amin(image_target) >= 0) and (np.amax(image_target) <= 1))
        cmax2 = np.amax(image_input)
        cmin2 = np.amin(image_input)
        image_input = rmin + ((image_input - cmin2) / (cmax2 - cmin2) * (rmax - rmin))
        assert ((np.amin(image_input) >= 0) and (np.amax(image_input) <= 1))
        mins = ((cmin1 + cmin2) / 2)
        maxs = ((cmax1 + cmax2) / 2)
        image_target = image_target.reshape((1, 512, 512))
        image_input = image_input.reshape((1, 512, 512))
        inputs_np = image_input
        targets_np = image_target

        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        # vgg_hq_b3 =  torch.from_numpy(vgg_hq_img3)
        # vgg_hq_b1 =  torch.from_numpy(vgg_hq_img1)
        #
        # vgg_hq_b3 = vgg_hq_b3.type(torch.FloatTensor)
        # vgg_hq_b1 = vgg_hq_b1.type(torch.FloatTensor)

        # print("hq vgg b3 {} b1 {}".format(vgg_hq_b3.shape , vgg_hq_b1.shape))
        self.sample = {'vol': input_file,
                       'HQ': targets,
                       'LQ': inputs,
                       # 'HQ_vgg_op':vgg_hq_b3, ## 1,256,56,56
                       # 'HQ_vgg_b1': vgg_hq_b1,  ## 1,256,56,56
                       'max': maxs,
                       'min': mins}
        return self.sample

#ENDS

# print("Main starts")
def main(args):
    # batch = 4
    # epochs = 50
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.allow_tf32 = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    # print("root dirs")
    # root_train_dir = "/projects/synergy_lab/garvit217/data_3d_2/train"
    # root_test_dir = "/projects/synergy_lab/garvit217/data_3d_2/test"
    # root_val_dir = "/projects/synergy_lab/garvit217/data_3d_2/validate"
    root_hq_dir = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_lq_dir = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"

    root_hq_dir_test = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    root_lq_dir_test = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"

    # data_loader = get_loader(mode=args.mode,
    #                          load_mode=args.load_mode,
    #                          saved_path=args.saved_path,
    #                          test_patient=args.test_patient,
    #                          patch_n=(args.patch_n if args.mode=='train' else None),
    #                          patch_size=(args.patch_size if args.mode=='train' else None),
    #                          transform=args.transform,
    #                          batch_size=(args.batch_size if args.mode=='train' else 1),
    #                          num_workers=args.num_workers)
    # print("dataset=CTdataset")
    # traindataset_ = VolCTDataset(root_train_dir)
    # testdataset_ = VolCTDataset(root_test_dir)
    # valdataset_ = VolCTDataset(root_val_dir)
    
    # print("data_loader = DataLoader")
    # train_loader = DataLoader(traindataset_, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=train_sampler)
    # test_loader = DataLoader(testdataset_, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=test_sampler)
    # val_loader = DataLoader(valdataset_, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=val_sampler)
    # train_loader = DataLoader(traindataset_, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # test_loader = DataLoader(testdataset_, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # val_loader = DataLoader(valdataset_, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # dataset_ = CTDataset(root_hq_dir, root_lq_dir,5120)
    # data_loader = DataLoader(dataset=dataset_, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # # print("Solver")
    # solver = Solver(args, data_loader)
    if args.mode == 'test':
        dataset_test = CTDataset(root_hq_dir_test, root_lq_dir_test, 914)  # dataset test length = 914
        data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        # For training, load the training dataset
        dataset_ = CTDataset(root_hq_dir, root_lq_dir, 5120)  # Adjust 5120 as needed for training
        data_loader = DataLoader(dataset=dataset_, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    solver = Solver(args, data_loader if args.mode == 'train' else data_loader_test)

    if args.mode == 'train':
        print("solver.train called here")
        solver.train()
        print("done training")
    elif args.mode == 'test':
        print("test called")
        solver.test()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--mode', type=str, default='train', help="train | test")
    # parser.add_argument('--load_mode', type=int, default=0, help="0 | 1")

    # # parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')
    # # parser.add_argument('--saved_path', type=str, default='./npy_img/')
    # parser.add_argument('--save_path', type=str, default='./save/')
    # parser.add_argument('--test_patient', type=str, default='L506')
    # parser.add_argument('--result_fig', type=bool, default=True)

    # parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    # parser.add_argument('--norm_range_max', type=float, default=3072.0)
    # parser.add_argument('--trunc_min', type=float, default=-160.0)
    # parser.add_argument('--trunc_max', type=float, default=240.0)

    # parser.add_argument('--transform', type=bool, default=False)
    # # if patch training, batch size is (--patch_n x --batch_size)
    # parser.add_argument('--patch_n', type=int, default=10)
    # # parser.add_argument('--patch_size', type=int, default=80)
    # parser.add_argument('--patch_size', type=int, default=64)

    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--num_epochs', type=int, default=100)

    # parser.add_argument('--print_iters', type=int, default=20)
    # parser.add_argument('--decay_iters', type=int, default=3000)

    # parser.add_argument('--save_iters', type=int, default=1000)
    # parser.add_argument('--test_iters', type=int, default=1000)

    # # parser.add_argument('--n_d_train', type=int, default=4)   

    # parser.add_argument('--lr', type=float, default=1e-5)#code has e-5 but paper says e-4??
    # # parser.add_argument('--lambda_', type=float, default=10.0) 

    # parser.add_argument('--device', type=str)
    # parser.add_argument('--num_workers', type=int, default=7)
    # parser.add_argument('--multi_gpu', type=bool, default=False)

    # args = parser.parse_args()
    # print("printing args")
    # print(args)
    # main(args)

    #Commented below this
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    # parser.add_argument('--data_path', type=str, default='/projects/synergy_lab/garvit217/data_3d_2/train')
    parser.add_argument('--saved_path', type=str, default='./npy_img3/')#same as below
    parser.add_argument('--save_path', type=str, default='./save5_batchsize32_lre4_epoch200/')# changed from save to save2
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-4) #trying e-4 now(From e-5) changing batch size

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
