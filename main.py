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

class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l,  length, root_hq_vgg3 = None, root_hq_vgg1= None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"


        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        self.img_list_l.sort()
        self.img_list_h.sort()

        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
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
        image_target = read_correct_image(self.data_root_h + self.img_list_h[idx])
        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])
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

    root_hq_dir = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_lq_dir = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"

    root_hq_dir_test = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    root_lq_dir_test = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"

    data_loader = None
    data_loader_test = None
    try:
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
    except  Exception as e:
        print(e)
if __name__ == "__main__":
    #Commented below this
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--load_mode', type=int, default=0)
    # # parser.add_argument('--data_path', type=str, default='/projects/synergy_lab/garvit217/data_3d_2/train')
    # parser.add_argument('--saved_path', type=str, default='./npy_img3/')#same as below
    parser.add_argument('--save_path', type=str, default='./weights')# changed from save to save2
    # parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    # parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    # parser.add_argument('--patch_n', type=int, default=10)
    # parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--print_iters', type=int, default=10)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)
    #trying e-4 now(From e-5) changing batch size

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
