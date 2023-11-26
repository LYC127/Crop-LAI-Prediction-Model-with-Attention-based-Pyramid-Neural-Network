import numpy as np
import cv2

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf


if __name__ == '__main__':
    a = cv2.imread(r'C:\Users\CS509\Desktop\33.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('s', a)
    cv2.waitKey(0)
    a = cv2.resize(a, (320, 320))
    a = a/255
    a = a.astype('uint8')
    a = onehot(a, 3)
    imgB = a.transpose(2, 0, 1)
    cv2.imshow('s1', imgB[1])
    cv2.waitKey(0)
