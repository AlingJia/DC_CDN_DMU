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
    # model = ResNet18_u()

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(r'D:\graduation\0-FAS-code\CDCN-master\DC_CDN_DMU\DC_CDN_DMU_train_result\DC_CDN_DMU_train_result_766.pkl'))

    print(model)


    for epoch in range(args.epochs):

        model.eval()

        with torch.no_grad():
            ###########################################
            '''                val             '''
            ###########################################
            # val for threshold
            val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir,
                                        transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

            map_score_list = []

            for i, sample_batched in enumerate(dataloader_val):
                # get the inputs
                inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                val_maps = sample_batched['val_map_x'].cuda()  # binary map from PRNet

                # pdb.set_trace()
                map_score = 0.0
                for frame_t in range(inputs.shape[1]):
                    mu, logvar, map_x = model(inputs[:, frame_t, :, :, :])
                    # score_norm = torch.sum(mu) / torch.sum(val_maps[:, frame_t, :, :])
                    score_norm = torch.mean(mu)
                    map_score += score_norm

                map_score = map_score / inputs.shape[1]

                map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
                # print(map_score_list)
                # pdb.set_trace()
            map_score_val_filename = args.log + '/' + args.log + '_map_score_val.txt'
            with open(map_score_val_filename, 'w') as file:
                file.writelines(map_score_list)
            print("把map_socer_val写入文件了")

            ###########################################
            '''                test             '''
            ##########################################
            # test for ACC
            test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir,
                                         transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

            map_score_list = []

            for i, sample_batched in enumerate(dataloader_test):
                # get the inputs
                inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                test_maps = sample_batched['val_map_x'].cuda()

                # pdb.set_trace()
                map_score = 0.0
                for frame_t in range(inputs.shape[1]):
                    mu, logvar, map_x = model(
                        inputs[:, frame_t, :, :, :])

                    # score_norm = torch.sum(mu) / torch.sum(test_maps[:, frame_t, :, :])
                    score_norm = torch.mean(mu)
                    map_score += score_norm

                map_score = map_score / inputs.shape[1]

                map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

            map_score_test_filename = args.log + '/' + args.log + '_map_score_test.txt'
            with open(map_score_test_filename, 'w') as file:
                file.writelines(map_score_list)

            #############################################################
            #       performance measurement both val and test
            #############################################################
            # val_threshold, best_test_threshold, val_ACC, val_ACER, val_APCER, val_BPCER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER = performances(map_score_val_filename, map_score_test_filename)

            # print('Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (
            #     epoch + 1, val_threshold, val_ACC, val_ACER))
            # log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (
            #     epoch + 1, val_threshold, val_ACC, val_ACER))
            #
            # print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
            #     epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            # log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (
            #     epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            # log_file.flush()
            val_threshold, best_test_threshold, val_ACC, val_ACER, val_APCER, val_BPCER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER = performances(
                map_score_val_filename, map_score_test_filename)

            log_file.write('val + test\n')
            log_file.write('Val: ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Best_test_threshold= %.4f\n' % (
            val_ACC, val_APCER, val_BPCER, val_ACER, val_threshold))
            log_file.write('Test:ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Best_test_threshold= %.4f\n' % (
            test_ACC, test_APCER, test_BPCER, test_ACER, best_test_threshold))
            log_file.flush()

            print("val + test")
            print('Val: ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Best_test_threshold= %.4f' % (
            val_ACC, val_APCER, val_BPCER, val_ACER, val_threshold))
            print('Test:ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Best_test_threshold= %.4f' % (
            test_ACC, test_APCER, test_BPCER, test_ACER, best_test_threshold))

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
    parser.add_argument('--log', type=str, default="result_447", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--test', action='store_true', default=True, help='')

    args = parser.parse_args()
    test()
