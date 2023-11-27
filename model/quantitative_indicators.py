'''
four measurement:
pixel accuracy
mean accuracy
mean IU
frequency weighted IU
'''
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def pixel_acc(x_data, y_label):
    count = 0
    for w in range(512):
        for h in range(512):
            if x_data[w][h] == y_label[w][h]:
                count += 1
    return count / (512.0*512.0)


def mean_acc(x_data, y_label):
    total_1 = 0
    total_2 = 0
    class_1 = 0
    class_2 = 0
    for w in range(512):
        for h in range(512):
            if y_label[w][h] == 0:
                total_1 += 1
            if y_label[w][h] == 255:
                total_2 += 1
            if x_data[w][h] == y_label[w][h]:
                if y_label[w][h] == 0:
                    class_1 += 1
                else:
                    class_2 += 1
    if total_2 == 0:
        return (class_1/total_1)
    else:
        return 0.5 * ((class_1/total_1) + (class_2/total_2))


def mean_IU(x_data, y_label):
    total_1, total_2, class_1, class_2 = 0, 0, 0, 0
    n2_1, n1_2 = 0, 0
    for w in range(512):
        for h in range(512):
            # ti 计算
            if y_label[w][h] == 0:
                total_1 += 1
            if y_label[w][h] == 255:
                total_2 += 1
            # nii计算
            if x_data[w][h] == y_label[w][h]:
                if y_label[w][h] == 0:
                    class_1 += 1
                else:
                    class_2 += 1
            # nji计算
            if y_label[w][h] == 255 and x_data[w][h] == 0:
                n2_1 += 1
            if y_label[w][h] == 0 and x_data[w][h] == 255:
                n1_2 += 1
    if (total_2 + n1_2) == 0:
        return (class_1/(total_1 + n2_1))
    else:
        return 0.5 * ((class_1/(total_1 + n2_1)) + (class_2/(total_2 + n1_2)))


def frequency_weighted_IU(x_data, y_label):
    total_1, total_2, class_1, class_2 = 0, 0, 0, 0
    n2_1, n1_2 = 0, 0
    for w in range(512):
        for h in range(512):
            # ti 计算
            if y_label[w][h] == 0:
                total_1 += 1
            if y_label[w][h] == 255:
                total_2 += 1
            # nii计算
            if x_data[w][h] == y_label[w][h]:
                if y_label[w][h] == 0:
                    class_1 += 1
                else:
                    class_2 += 1
            # nji计算
            if y_label[w][h] == 255 and x_data[w][h] == 0:
                n2_1 += 1
            if y_label[w][h] == 0 and x_data[w][h] == 255:
                n1_2 += 1
    if (total_2 + n1_2) == 0:
        return (1/(512*512)) * ((total_1*class_1 / (total_1 + n2_1)))
    else:        
        return (1/(512*512)) * ((total_1*class_1 / (total_1 + n2_1)) + (total_2*class_2 / (total_2 + n1_2)))


if __name__ == '__main__':
    res = []
    img_name = os.listdir('../datasets/semi_test/images/')
    for i in tqdm(img_name):
        label_path = '../datasets/semi_test/labels/{}'.format(i)
        # HSV_predict_path = 'HSV_predict/{}.png'.format(i)
        FCN_predict_path = 'result/{}'.format(i)
        # OTSU_predict_path = 'OTSU_predict/{}.png'.format(i)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (512, 512))
        label = label.astype(np.float32)
        # HSV_predict = cv2.imread(HSV_predict_path, cv2.IMREAD_GRAYSCALE)
        # HSV_predict = HSV_predict.astype(np.float32)
        FCN_predict = cv2.imread(FCN_predict_path, cv2.IMREAD_GRAYSCALE)
        FCN_predict = FCN_predict.astype(np.float32)
        # OTSU_predict = cv2.imread(OTSU_predict_path, cv2.IMREAD_GRAYSCALE)
        # OTSU_predict = OTSU_predict.astype(np.float32)

        # pixel accuracy
        # hsv_p_acc = pixel_acc(HSV_predict, label)
        fcn_p_acc = pixel_acc(FCN_predict, label)
        # otsu_p_acc = pixel_acc(OTSU_predict, label)

        # mean accuracy
        # hsv_m_acc = mean_acc(HSV_predict, label)
        fcn_m_acc = mean_acc(FCN_predict, label)
        # otsu_m_acc = mean_acc(OTSU_predict, label)

        # mean IU
        # hsv_m_IU = mean_IU(HSV_predict, label)
        fcn_m_IU = mean_IU(FCN_predict, label)
        # otsu_m_IU = mean_IU(OTSU_predict, label)

        # frequency weighted IU
        # hsv_f_IU = frequency_weighted_IU(HSV_predict, label)
        fcn_f_IU = frequency_weighted_IU(FCN_predict, label)
        # otsu_f_IU = frequency_weighted_IU(OTSU_predict, label)

        # res.append([str(i) + '.png', hsv_p_acc, fcn_p_acc, otsu_p_acc,
        #             hsv_m_acc, fcn_m_acc, otsu_m_acc,
        #             hsv_m_IU, fcn_m_IU, otsu_m_IU,
        #             hsv_f_IU, fcn_f_IU, otsu_f_IU])
        res.append([i, fcn_p_acc, fcn_m_acc, fcn_m_IU, fcn_f_IU])
    # for i in range(len(res)):
    #     print(i, res[i][0], res[i][1], res[i][2], res[i][3])
    df = pd.DataFrame(data=res)
    df.columns = ['图片', 'unet_p_acc', 'unet_m_acc',  'unet_m_IU', 'unet_f_IU']
    df.to_excel('hed_fcn2s.xlsx', index=False)



