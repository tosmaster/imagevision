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
import six

qd_names =['cup','garden hose', 'marker', 'truck', 'oven', 'cooler', 'birthday cake',
'camouflage', 'pool', 'dog', 'bear','bird', 'The Great Wall of China','van',
'tiger', 'bench', 'hot tub','coffee cup', 'telephone', 'mug','matches',
'animal migration', 'lantern', 'skyscraper','keyboard','foot','monkey','sleeping bag',
'brain', 'peanut', 'belt', 'tent','cookie', 'sweater','hot dog',
'microwave', 'mermaid', 'donut', 'hourglass', 'bee']

STROKE_COUNT = 196

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    #Unwrap the list
    in_strokes = [(xi,yi,i) for i,(x,y) in enumerate(stroke_vec) for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    
    #Replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    
    #Pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)

def read_batch(samples=5, start_row = 0):
    """
    load and process the csv files
    this function is horribly inefficient but simple
    """
    out_df_list = []
    for c_path in ALL_TRAIN_PATHS:
        c_df = pd.read_csv(c_path, nrows=samples, skiprows=start_row)
        #print(type(c_df), ";")
        c_df.columns=COL_NAMES
        out_df_list += [c_df[['drawing', 'word']]]
        #print(out_df_list, "==")
    full_df = pd.concat(out_df_list)
    full_df['drawing'] = full_df['drawing'].\
        map(_stack_it)
    
    return full_df


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
        strokes = df['drawing'].map(_stack_it)
        image = entropybag.compute()
        self.dataset = np.array(list(zip(image,strokes,df['y'])))
        self.total_count = len(self.dataset)
        self.transforms = transforms
        self.image_size = image_size
        print("No = {} and total number of items {}".format(no,self.total_count))


    def __getitem__(self, index):
        img,stroke,label = self.dataset[index]
        
        if self.transforms is not None:
        	img = np.asarray(img).reshape(self.image_size[0],self.image_size[1]).astype('uint8')
        	img = Image.fromarray(img)
        	img = self.transforms(img)
        	img = img * 256
        img = img.reshape(1,self.image_size[0],self.image_size[1])
        return img,stroke,label

    def __len__(self):
        return self.total_count