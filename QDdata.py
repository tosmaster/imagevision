import os
import torch
import urllib
import cv2
from dask import bag
import random
from ast import literal_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from skimage.draw import line_aa


qd_names =['cup','garden hose', 'marker', 'truck', 'oven', 'cooler', 'birthday cake',
'camouflage', 'pool', 'dog', 'bear','bird', 'The Great Wall of China','van',
'tiger', 'bench', 'hot tub','coffee cup', 'telephone', 'mug','matches',
'animal migration', 'lantern', 'skyscraper','keyboard','foot','monkey','sleeping bag',
'brain', 'peanut', 'belt', 'tent','cookie', 'sweater','hot dog',
'microwave', 'mermaid', 'donut', 'hourglass', 'bee']

# The following functions come from Sketch-A-Net
# [Sketch-A-XNORNet](http://github.com/ayush29feb/Sketch-A-XNORNet)
def data_draw_cv2(raw_strokes, size=96, linewidth=6, time_color=True):
    img = np.zeros((256, 256), np.uint8)
    for t, stroke in enumerate(literal_eval(raw_strokes)):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), 
                         (stroke[0][i + 1], stroke[1][i + 1]), color, linewidth)
    if size != 256:
        img = cv2.resize(img, (size, size))
        
    img = np.array(img)
    return img

def get_bounds(strokes):
	"""Given a 3-stroke sequence returns the bounds for the respective sketch

	Args:
		strokes: A 3-stroke sequence representing a single sketch

	Returns:
		(min_x, max_x, min_y, max_y): bounds of the sketch
	"""
	min_x, max_x, min_y, max_y = (0, 0, 0, 0)
	abs_x, abs_y = (0, 0)

	for i in range(strokes.shape[0]):
		dx, dy = strokes[i, :2]
		abs_x += dx
		abs_y += dy
		min_x = min(min_x, abs_x)
		max_x = max(max_x, abs_x)
		min_y = min(min_y, abs_y)
		max_y = max(max_y, abs_y)

	return (min_x, max_x, min_y, max_y)

def strokes_to_npy(strokes):
	"""Given a 3-stroke sequence returns the sketch in a numpy array

	Args:
		strokes: A 3-stroke sequence representing a single sketch

	Returns:
		img: A grayscale 2D numpy array representation of the sketch
	"""
	min_x, max_x, min_y, max_y = get_bounds(strokes)
	dims = (50 + max_x - min_x, 50 + max_y - min_y)
	img = np.zeros(dims, dtype=np.uint8)
	abs_x = 25 - min_x
	abs_y = 25 - min_y
	pen_up = 1
	for i in range(strokes.shape[0]):
		dx, dy = strokes[i, :2]

		if pen_up == 0:
			rr, cc, val = line_aa(abs_x, abs_y, abs_x + dx, abs_y + dy)
			img[rr, cc] = val * 255

		abs_x += dx
		abs_y += dy
		pen_up = strokes[i, 2]

	# TODO: Why do we need to transpose? Fix get_bounds accordingly
	return img.T

def reshape_to_square(img, size=225):
	"""Given any size numpy array return

	Args:
		img: A grayscale 2D numpy array representation of the sketch

	Returns:
		img_sq: A grayscale 2D numpy array representation of the sketch fitted
				in a size x size square.
	"""
	# TODO: make sure this formula is correct
	# TODO: draw in a square instead of rescaling
	img_resize = imresize(img, float(size) / max(img.shape))
	w_, h_ = img_resize.shape
	x, y = ((size - w_) / 2, (size - h_) / 2)

	img_sq = np.zeros((size, size), dtype=np.uint8)
	img_sq[x:x + w_, y:y + h_] = img_resize

	return img_sq

class QDStrokeDataset(data.Dataset):
    def __init__(self, class_names = qd_names, data_dir = "./input/train_simplified", output = "output",
                 start = 0, count = 340 * 2048,images_category = 2048,image_size = (256,256),transforms=None):

        print("Total class number:{}".format(len(class_names)))
     
        total = 0
        whole_data = np.zeros((count, image_size[0]*image_size[1]),dtype = np.uint8)
        label = np.zeros(count,dtype=np.int32)
        
        for i,name in enumerate(tqdm(class_names)):
                full_name = data_dir + "/" + name +".csv"
                if (os.path.isfile(full_name) == False):
                    print("Error! {} does not exist.".format(full_name))

                cat_data = pd.read_csv(full_name)['drawing']
                
                each_cat = int(count//len(class_names))
                if each_cat > images_category:
                    each_cat = images_category
                    
                if each_cat > len(cat_data):
                    each_cat = len(cat_data)
                    
                end = start + images_category
                if end > len(cat_data):
                    end = cat_data
                    print("images_category is larger than the length of .npz")
                    
                if each_cat > end - start:
                    each_cat = end - start
                    print(class_names[i]," only gets ",each_cat)
                          
                mask = range(start, end)
                mask = np.random.choice(mask,size = each_cat)
                
                for j,item in enumerate(cat_data[mask]):
                    item = reshape_to_square(strokes_to_npy(item),image_size[0])
                    whole_data[total + j] = item
                    label[total + j] = i
                del cat_data
                
                total += each_cat
                
        self.class_names = class_names
        self.whole_data = whole_data[:total]
        self.label = label[:total]
        self.dataset = list(zip(whole_data[:total],label[:total]))
        self.total_count = total
        self.image_size = image_size
        self.transforms = transforms
        
        print("Total number of items:",self.total_count,len(self.dataset))


    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transforms is not None:
        	img = np.asarray(img).reshape(self.image_size[0],self.image_size[1]).astype('uint8')
        	img = Image.fromarray(img)
        	img = self.transforms(img)
        	img = img * 256
        img = img.reshape(1,self.image_size[0],self.image_size[1])
        return img,label

    def __len__(self):
        return self.total_count	

class QDDetection(data.Dataset):

    def __init__(self, class_names = qd_names, data_dir = "./input", output = "output",
                 start = 0, count = 340 * 2048,images_category = 2048,image_size = (28,28),transforms=None):

        print("Total class number:{}".format(len(class_names)))
     
        total = 0
        whole_data = np.zeros((count, image_size[0]*image_size[1]),dtype = np.uint8)
        label = np.zeros(count,dtype=np.int32)
        
        for i,name in enumerate(tqdm(class_names)):
                if (os.path.isfile("./pic28/{}.npy".format(name)) == False):
                    print("Error! {} does not exist.".format(name))

                cat_data = np.load("./pic28/{}.npy".format(name))
                
                each_cat = int(count//len(class_names))
                if each_cat > images_category:
                    each_cat = images_category
                    
                if each_cat > len(cat_data):
                    each_cat = len(cat_data)
                    
                end = start + images_category
                if end > len(cat_data):
                    end = cat_data
                    print("images_category is larger than the length of .npz")
                    
                if each_cat > end - start:
                    each_cat = end - start
                    print(class_names[i]," only gets ",each_cat)
                          
                mask = range(start, end)
                mask = np.random.choice(mask,size = each_cat)
                
                for j,item in enumerate(cat_data[mask]):
                    whole_data[total + j] = item
                    label[total + j] = i
                del cat_data
                
                total += each_cat
                
        self.class_names = class_names
        self.whole_data = whole_data[:total]
        self.label = label[:total]
        self.dataset = list(zip(whole_data[:total],label[:total]))
        self.total_count = total
        self.image_size = image_size
        self.transforms = transforms
        
        print("Total number of items:",self.total_count,len(self.dataset))


    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transforms is not None:
        	img = np.asarray(img).reshape(self.image_size[0],self.image_size[1]).astype('uint8')
        	img = Image.fromarray(img)
        	img = self.transforms(img)
        	img = img * 256
        img = img.reshape(1,self.image_size[0],self.image_size[1])
        return img,label

    def __len__(self):
        return self.total_count

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

    def __init__(self,no = 0, data_file="data",label_file="label",image_size=(96,96),transforms=None):
        
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
        self.transforms = transforms
        
        print("Total number of items:",self.total_count)


    def __getitem__(self, index):
        img = self.dataset[index]
        label = self.label[index]
        if self.transforms is not None:
        	img = np.asarray(img).reshape(self.image_size[0],self.image_size[1]).astype('uint8')
        	img = Image.fromarray(img)
        	img = self.transforms(img)
        	img = img * 256
        img = img.reshape(1,self.image_size[0],self.image_size[1])
        return img,label

    def __len__(self):
        return self.total_count
    
class QDloadStrokeData(data.Dataset):

    def __init__(self,no = 0, data_file=None,val = False, image_size=(96,96),transforms=None):
        
        if data_file == None:
            if val == True:
                data_file = "./val/val_dataset.csv"
            else:
                data_file = "./train/" + 'train_k{}.csv.gz'.format(no)
                
        if os.path.exists(data_file) == False:
            print(data_file,"does not exist\n")
            
        df = pd.read_csv(data_file)
        entropybag = bag.from_sequence(df.drawing.values).map(data_draw_cv2)
        image = entropybag.compute()
        self.dataset = np.array(list(zip(image,df['y'])))
        self.total_count = len(self.dataset)
        self.transforms = transforms
        self.image_size = image_size
        print("No = {} and total number of items {}".format(no,self.total_count))


    def __getitem__(self, index):
        img,label = self.dataset[index]
        
        if self.transforms is not None:
        	img = np.asarray(img).reshape(self.image_size[0],self.image_size[1]).astype('uint8')
        	img = Image.fromarray(img)
        	img = self.transforms(img)
        	img = img * 256
        img = img.reshape(1,self.image_size[0],self.image_size[1])
        return img,label

    def __len__(self):
        return self.total_count