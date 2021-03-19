import numpy as np
from numpy import random
import scipy
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
import os
#import random



def get_tensor(dl_train):
    data = []
    for i, j in enumerate(dl_train):
        digits = {'index': i, 'pix': j}
        data.append(digits)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return npimg

if __name__ == '__main__':


    train_set = datasets.MNIST('./data', train=True, download=True)
    train_set_array = train_set.data.numpy()
    #print(train_set_array[0])
    #print(np.ndim(train_set_array[0]))
    labels = train_set.targets
    idx = train_set.train_labels==9
    labels_for_growspace = train_set.train_labels[idx]
    mnist_growspace = train_set.train_data[idx]
    path = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/mnist_data/9/'
    good = mnist_growspace.data.numpy()
    length = len(good)
    samples = np.random.randint(length-1, size=50)
    print(samples)
    #print('what is lenght:',len(good))
    # im1 = np.zeros((28, 28), dtype=np.uint8)
    # im2 = np.zeros((28, 28), dtype=np.uint8)
    # im3 = np.zeros((28, 28, 3), dtype=np.uint8)
    # number = np.where(good[90] == 0, im2, 1 * 255)
    # # print(number)
    # # print(np.ndim(number))
    # img = np.dstack((im1, im2, number))
    # A = np.array((im3, img))
    # final_img = np.float32(np.sum(A, axis=0))
    # cv2.imshow('mnist', final_img)
    # cv2.waitKey(-1)

    # #print(labels[0])
    #
    # number = random.choice(good, 100)
    # cv2.imshow('test',number[1])
    # cv2.waitKey(-1)
    #     im4 = []
    #     im1 = np.zeros((28, 28), dtype=np.uint8)
    #     im2 = np.zeros((28, 28), dtype=np.uint8)
    #     im3 = np.zeros((28, 28, 3), dtype=np.uint8)
    #     number = np.where(number[x] == 0, im2, 1 * 255)
    #     #print(number)
    #     #print(np.ndim(number))
    #     img = np.dstack((im1,im2, number))
    #     A = np.array((im3, img))
    #     final_img = np.float32(np.sum(A, axis=0))
    #     cv2.imshow('mnist', final_img)
    #print(number[1])
    s = 1
    for i in samples:
        #if x == 2:
        im4 = []
        im1 = np.zeros((28, 28), dtype=np.uint8)
        im2 = np.zeros((28, 28), dtype=np.uint8)
        im3 = np.zeros((28, 28, 3), dtype=np.uint8)
        number = np.where(good[i] == 0, im2, 1 * 255)
        #print(number)
        #print(np.ndim(number))
        img = np.dstack((im1,im2, number))
        A = np.array((im3, img))
        final_img = np.float32(np.sum(A, axis=0))
        #cv2.waitKey(-1)
        cv2.imwrite(os.path.join(path, '9_'+ str(s)+'.png'), final_img)
        s+=1
    # train_set_array = train_set.data.numpy()
    # #print(train_set_array[0].target)
    # data_iter = iter(train_set_array)
    # images, labels = data_iter.next()
    # print(labels)
    # yellow = (0, 128, 128)  # RGB color (dark yellow)
    # ## create empty
    # im1 = np.zeros((28,28),dtype = np.uint8)
    # im2 = np.zeros((28,28),dtype = np.uint8)
    # #number = np.where(train_set_array[0]==0,im1,1*255)
    # im3 = np.zeros((28,28,3), dtype = np.uint8)
    # cv2.rectangle(
    #     im3, pt1=(7, 0), pt2=(12, 28), color=yellow, thickness=-1)
    #
    # number = np.where(train_set_array[0] == 0, im1, 1 * 255)
    # img = np.dstack((im1,im2, number))
    # A = np.array((im3, img))
    # final_img = np.float32(np.sum(A, axis= 0))
    # #print(img)
    # #cv2.drawKeypoints(im3,number,im3)
    # #img5 = np.sum((im3, img), axis = 0)
    # #print(img5)
    # #img = cv2.flip(final,0)
    # #cv2.imshow('mnist', img)
    # cv2.line(final_img, (10,28), (10, 23), (0,255,0), 1)
    # #yellow = (0, 128, 128)  # RGB color (dark yellow)
    # #cv2.rectangle(
    #     #img, pt1=(7, 0), pt2=(12, 28), color=yellow, thickness=-1)
    # # print(img.shape)
    # path = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/mnist_data'
    #
    # #res = cv2.resize(img*255,dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    # #cv2.imshow('originalmnist',train_set_array[0])
    # cv2.imshow('mnist',final_img)
    # #cv2.imwrite(os.path.join(path, 'mnist_digit_original.png'), res)
    # cv2.waitKey(-1)