import cv2
import os


if __name__ == '__main__':
    for file in os.listdir('./result'):
        if file == '.DS_Store':
            continue
        print(file)
        image = cv2.imread('./result/' + file)
        cv2.imwrite('./result/' + file, 255-image)
