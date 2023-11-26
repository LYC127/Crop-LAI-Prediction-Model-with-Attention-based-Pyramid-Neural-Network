import torch
import cv2
import numpy as np
from torchvision import transforms
import visdom
# import pandas as pd
import csv



if __name__ == '__main__':
    # pic_path = '1.jpg'

    device = torch.device("cuda")
    model = torch.load('checkpoints_fcn2s_adam_1e3_All/fcn_model_200.pt')
    model.to(device)
    model.eval()
    LAI = []
    for i in range(0,82):
        pic_path = 'data/'+ str(i) +'.png'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        imgA = cv2.imread(pic_path)
        imgA = cv2.resize(imgA, (320, 320))
        imgA = transform(imgA)

        # ----
        img = torch.unsqueeze(imgA, 0)
        img = img.to(device)

        #  -------
        output = model(img)
        output = torch.sigmoid(output)

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)

        # vis = visdom.Visdom()
        # vis.images(output_np[:, None, :, :], win='pred', opts=dict(title='prediction'))

        shape = output_np[:, None, :, :][0][0].shape
        LAI.append(np.sum(output_np[:, None, :, :][0][0] == 0) / (shape[0]*shape[1]))
        # print(np.sum(output_np[:, None, :, :][0][0] == 0) / (shape[0]*shape[1]))

    LAI = list(map(lambda x: [x], LAI))
    print(LAI)
    csvFile = open('LAI.csv', "w+")
    try:
        writer = csv.writer(csvFile)
        writer.writerow("LAI")
        for i in range(len(LAI)):
            writer.writerow(LAI[i])
    finally:
        csvFile.close()

