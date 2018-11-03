import os
import torch
import urllib
import cv2
import random
from ast import literal_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils import data
import matplotlib.pyplot as plt

qd_names =['clock', 'birthday cake','hamburger', 'eyeglasses', 'basketball', 'rake', 'string bean', 'dragon', 
	'sea turtle','stove', 'table', 'windmill', 'bird', 'star', 'crocodile', 'mountain', 
        'pig', 'owl', 'cake', 'harp', 'cup', 'tooth', 'camouflage', 'lollipop',
        'remote control', 'blueberry', 'grass', 'floor lamp', 'dresser', 'sheep','foot', 'rain', 
	'sink', 'bat', 'pineapple', 'violin', 'postcard','firetruck', 'lightning', 'broccoli']

class QDloadtest(data.Dataset):

    def __init__(self,class_names = qd_names,name = "./test/test_simplified.npy",image_size = (28,28)):

        if (os.path.isfile(name) == False):
            print("Error! {} does not exist.".format(name))
            return
        if name.endswith(".npz"):
            cat_data = np.load(name)["arr_0"]
        else:
            cat_data = np.load(name)
        
        self.class_names = class_names
        self.dataset = cat_data
        self.total_count = len(cat_data)
        self.image_size = image_size
        
        print("Total number of items:",self.total_count)


    def __getitem__(self, index):
        img = self.dataset[index]
        return img.reshape(1,self.image_size[0],self.image_size[1])

    def __len__(self):
        return self.total_count

class QDcreateData():
    
    def create(self,class_names=qd_names, start=0, length=3000, dir_name="./pic96",image_size=(96,96)):
        
        whole_data = np.zeros((len(class_names) * length, image_size[0]*image_size[1]),dtype = np.uint8)
        label = np.zeros(len(class_names) * length,dtype = np.int32)

        for i,name in enumerate(tqdm(class_names)):
            file_name = os.path.join(dir_name,name)
            file_name += ".npz"
            each_data = np.load(file_name)["arr_0"]
            each_data = each_data[start:start+length]
            for j in range(length):
                label[i*length+j] = i
                whole_data[i*length+j] = each_data[j]
                    
            del each_data
 
        mask = list(range(len(class_names) * length)) 
        random.shuffle(mask)
        whole_data = whole_data[mask]
        label = label[mask]
        
        print(label[:3])
        
        no = start//length
        np.save("data"+str(no)+".npy",whole_data)
        np.save("label"+str(no)+".npy",label)
        del whole_data
        del label
            
        
class QDloadData(data.Dataset):

    def __init__(self,no = 0, data_file="data",label_file="label",image_size=(96,96)):
        
        data_name = data_file.replace(".npy","")
        label_file = label_file.replace(".npy","")
        data_name = data_file + str(no)+".npy"
        label_name = label_file + str(no)+".npy"
        print(data_name,label_name)
        if (os.path.isfile(data_name) == False or os.path.isfile(label_name) == False):
            print("Error! {} does not exist.".format(data_file))
            return

        image_data = np.load(data_name)
        label_data = np.load(label_name)
        self.dataset = image_data
        self.label = label_data
        self.total_count = len(image_data)
        self.image_size = image_size
        
        print("Total number of items:",self.total_count)


    def __getitem__(self, index):
        img = self.dataset[index]
        label = self.label[index]
        return img.reshape(1,self.image_size[0],self.image_size[1]),label

    def __len__(self):
        return self.total_count
    
