from __future__ import print_function, division
import torch

torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import argparse, os
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms

from DC_CDN_DMU.models.DC_CDN_DUM import DC_CDN_DMU

from DC_CDN_DMU.load_DC_CDN_DMU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

import torch.optim as optim

from DC_CDN_DMU.utils import performances

# Dataset root

train_image_dir = r"D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-Crop\Train_files"
val_image_dir = r'D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-Crop\Dev_files'
test_image_dir = r'D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-Crop\Test_files'

train_map_dir = r"D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-depth\Train_files"
val_map_dir = r"D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-depth\Dev_files"
test_map_dir = r"D:\graduation\0-FAS-code\OULU-NPU-processed\OULU-frame-depth\Test_files"

train_list = r"D:\graduation\0-FAS-code\OULU-NPU-otherfile\Protocols-delete0to10\Protocol_1\Train.txt"
val_list = r"D:\graduation\0-FAS-code\OULU-NPU-otherfile\Protocols-delete0to10\Protocol_1\Dev.txt"
test_list = r"D:\graduation\0-FAS-code\OULU-NPU-otherfile\Protocols-delete0to10\Protocol_1\Test.txt"

test_one = r"D:\graduation\0-FAS-code\OULU-NPU-otherfile\Protocols-delete0to10\Protocol_1\Dev.txt"
# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap(x,  mu,logvar,map_x,label, str):
    label = label[0].item()

    ## initial images
    feature_first_frame = x[0, :, :, :].cpu()  ## the middle frame

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    name = '{}_x_visual{}.jpg'.format(str,label)
    plt.savefig(args.log + '/' + name)
    plt.close()



    ## depthmap
    heatmap2 = torch.pow(map_x[0, :, :], 2)  ## the middle frame

    heatmap2 = heatmap2.data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    name ='{}_x_depmap{}.jpg'.format(str, label)

    plt.savefig(args.log + '/' + name)
    plt.close()


    ## mu
    heatmap3 = torch.pow(mu[0, :, :], 2)  ## the middle frame

    heatmap3 = heatmap3.data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap3)
    plt.colorbar()
    name ='{}_x_mu{}.jpg'.format(str, label)

    plt.savefig(args.log + '/' + name)
    plt.close()


    ## logvar
    heatmap4 = torch.pow(logvar[0, :, :], 2)  ## the middle frame

    heatmap4 = heatmap4.data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap4)
    plt.colorbar()
    name ='{}_x_logvar{}.jpg'.format(str, label)

    plt.savefig(args.log + '/' + name)
    plt.close()

# main function
def test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_test_log_P1.txt', 'w')

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    print('test!\n')
    log_file.write('test!\n')
    log_file.flush()

    model = DC_CDN_DMU()
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(r'D:\graduation\0-FAS-code\CDCN-master\DC_CDN_DMU\DC_CDN_DMU_train_result\DC_CDN_DMU_train_result_766.pkl'))


    for epoch in range(args.epochs):

        model.eval()

        with torch.no_grad():

            ###########################################
            '''                test             '''
            ##########################################
            # test for ACC
            test_data = Spoofing_valtest(test_one, val_image_dir, val_map_dir,
                                         transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

            map_score_list = []

            for i, sample_batched in enumerate(dataloader_test):
                # get the inputs
                inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                test_maps = sample_batched['val_map_x'].cuda()
                image_names = sample_batched['image_names']

                # pdb.set_trace()
                map_score = 0.0
                # for frame_t in range(inputs.shape[1]):
                mu, logvar, embedding= model(inputs[:, 0, :, :, :])
                FeatureMap2Heatmap(inputs[:, 0, :, :, :],mu,logvar,embedding, spoof_label, image_names[0][0])

            #         # score_norm = torch.sum(mu) / torch.sum(test_maps[:, frame_t, :, :])
            #         map_score_norm = torch.mean(map_x)
            #         map_score += map_score_norm
            #
            #
            #     map_score = map_score / inputs.shape[1]
            #
            #     map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
            #
            # map_score_test_filename = args.log + '/' + args.log + '_map_score_test.txt'
            # with open(map_score_test_filename, 'w') as file:
            #     file.writelines(map_score_list)



    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--kl_lambda', type=float, default=0.001, help='')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
    parser.add_argument('--log', type=str, default="tupian_all", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--test', action='store_true', default=True, help='')

    args = parser.parse_args()
    test()
