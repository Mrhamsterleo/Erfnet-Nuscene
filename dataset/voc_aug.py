import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Default encoding for pixel value, class name, and class color
class_name = {0: 'human',
      1: 'vehicle',
      2: 'movable_object',
      3: 'background',
      4: 'other'}


color_encoding = {0: (255, 0, 0),
      1: (255, 255, 0),
      2: (0, 0, 255),
      3: (0, 255, 0),
      4: (255, 0, 255)}

def rgb_to_label(rgb_image, colormap = color_encoding):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]
    encoded_image = np.zeros(shape,dtype=np.int8)
    for i, cls in enumerate(colormap):
        for x in range(encoded_image.shape[0]):
            for y in range (encoded_image.shape[1]):
                if(np.all(rgb_image[x][y] == colormap[i])):
                    encoded_image[x][y] = i
    return encoded_image


def label_to_rgb(label, colormap = color_encoding):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    output = np.zeros(label.shape[:2]+(3,))
    for k in colormap.keys():
        output[label==k] = colormap[k]
    return np.uint8(output)

class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/content/drive/My Drive/internship/ERFNet-CULane-PyTorch/list', data_list='train', transform=None):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            # self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                # self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                # self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                self.img_list.append(dataset_path + line.strip().split(" ")[0])
                self.label_list.append(dataset_path + line.strip().split(" ")[1])
                # self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img' # 'val'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        # image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        # label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(self.img_list[idx]).astype(np.float32)
        label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
        # label = rgb_to_label(label, colormap=color_encoding)
        # exist = self.exist_list[idx]
        # image = image[240:, :, :]
        # label = label[240:, :]
        label = label.squeeze()
        if self.transform:
            image, label = self.transform((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
        if self.is_testing:
            return image, label #, self.img[idx]
        else:
            return image, label #, exist
