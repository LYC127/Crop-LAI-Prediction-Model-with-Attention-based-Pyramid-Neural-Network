from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import os

from BagData import test_dataloader, train_dataloader
from FCN import FCN2s, FCN8s, newVGG16, VGG16
#from newHED_FCN2s import FCN2s
from quantitative_indicators import pixel_acc, mean_acc, mean_IU, frequency_weighted_IU
from loss import Loss
# UNet
from unet import UNet
from torchsummary import summary

import pytorch_ssim
import pytorch_iou
#

def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")

def get_para_num(model):
    lst = []
    for para in model.parameters():
        lst.append(para.nelement())
    print(f"total paras number: {sum(lst)}")

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
vgg_model = newVGG16()
fcn_model = FCN2s(pretrained_net=vgg_model, n_class=1)
fcn_model = fcn_model.to(device)
#total_trainable_params = sum(p.numel() for p in fcn_model.parameters() if p.requires_grad)
#print(f'{total_trainable_params:,} training parameters.')
print(fcn_model)
#
#vgg_model = newVGG16()
#unet_model = UNet(n_channels = 3, n_classes = 2, pretrained_net = vgg_model)
#print(unet_model)
#fcn_model = unet_model.to(device)
for para in fcn_model.pretrained_net.parameters():
   para.requires_grad = False
get_trainable_para_num(fcn_model)
get_para_num(fcn_model)

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss


def train(epo_num=50, show_vgg_params=False):

    vis = visdom.Visdom(use_incoming_socket=False)

    #
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # UNet
    # criterion = Loss(True, True).to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = nn.BCELoss().to(device)


    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-3)
    # fcn adam 1e-3
    # optimizer = optim.RMSprop(fcn_model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    best_testloss = 100
    for epo in range(epo_num):
        
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            # fuse = torch.sigmoid(fuse)
            # loss1 = criterion(output, bag_msk.float())
            # loss2 = weighted_cross_entropy_loss(fuse, edge)
            # loss = loss1 + loss2
            loss = bce_ssim_loss(output, bag_msk).to(device)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()


            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                # vis.images(output_np[0][0], win='train_pred', opts=dict(title='train prediction'))
                # vis.images(bag_msk_np[0][0], win='train_label', opts=dict(title='train label'))
                # vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

        test_loss = 0
        fcn_model.eval()
        fcn_p_acc = 0
        fcn_m_acc = 0
        fcn_m_IU = 0
        fcn_f_IU = 0
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                # fuse = torch.sigmoid(fuse)
                # loss1 = criterion(output, bag_msk.float())
                # loss2 = weighted_cross_entropy_loss(fuse, edge)
                # loss = loss1 + loss2
                # loss = criterion(output, bag_msk)
                loss = bce_ssim_loss(output, bag_msk).to(device)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss
                
                # softmax = nn.Softmax(dim=1)
                # output = softmax(output)
                # output = torch.argmax(output, dim=1)
                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)
                # output_np = output_np[0]
                output_np = (output_np[0][0] > 0.5).astype(np.int32)
                # output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = bag_msk_np[0][0]
                #print(output_np.shape, bag_msk_np.shape)
                
                if (np.mod(epo, 10) == 0):
                    fcn_p_acc += pixel_acc(output_np*255, bag_msk_np*255)
                    fcn_m_acc += mean_acc(output_np*255, bag_msk_np*255)
                    fcn_m_IU += mean_IU(output_np*255, bag_msk_np*255)
                    fcn_f_IU += frequency_weighted_IU(output_np*255, bag_msk_np*255)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    # vis.images(output_np[0][0], win='test_pred', opts=dict(title='test prediction'))
                    # vis.images(bag_msk_np[0][0], win='test_label', opts=dict(title='test label'))
                    # vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

            fcn_p_acc = fcn_p_acc/(index+1)
            fcn_m_acc = fcn_m_acc/(index+1)
            fcn_m_IU = fcn_m_IU/(index+1)
            fcn_f_IU = fcn_f_IU/(index+1)
            print('fcn_p_acc = %f, fcn_m_acc = %f, fcn_m_IU = %f, fcn_f_IU = %f'
                  %(fcn_p_acc, fcn_m_acc, fcn_m_IU, fcn_f_IU))

        with open(os.path.join("./", "model_fcn2s_adam_1e3_All.log"), 'a+') as f:
            f.write("------------------" + "Epoch: " + str(epo) + "------------------" + '\n')
            f.write("Train_loss = " + str(train_loss) + '\n')
            f.write("Test_loss = " + str(test_loss) + '\n')
            f.write("fcn_p_acc = " + str(fcn_p_acc) + '\n')
            f.write("fcn_m_acc = " + str(fcn_m_acc) + '\n')
            f.write("ffcn_m_IU = " + str(fcn_m_IU) + '\n')
            f.write("fcn_f_IU = " + str(fcn_f_IU) + '\n')

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints_fcn2s_adam_1e3_All/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints_fcn2s_adam_1e3_All/fcn_model_{}.pt'.format(epo))

        #if test_loss/len(test_dataloader) < best_testloss:
            #best_testloss = test_loss/len(test_dataloader)
            #torch.save(fcn_model, 'checkpoints_fcn2s_adam_1e3_Cucumber/fcn_best_model.pt')
            #print('saveing checkpoints_fcn2s_adam_1e3_Cucumber/fcn_best_model.pt')


if __name__ == "__main__":

    train(epo_num=501, show_vgg_params=False)
    torch.save(fcn_model.state_dict(), 'net_params.pkl')

