import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.transforms as _transform
import torch
import tifffile as tf
try:
    import transform as T
except:
    import data_utils.transform as T

traindir = "train"
testdir = "test"
imagedir = 'images'
labeldir = 'labels_0-1'

channel_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
#channel_list = ['1-C11', '2-C12_real', '3-C12_imag', '4-C22', '5-alpha', '6-anisotropy', '7-entropy']
trainfile = r'E:\nxy\net\Data\train\labels_0-1'
testfile = r"E:\nxy\net\Data\test\labels_0-1"



# mean_train = np.array([0.032213421442935264, 0.03168874880981447, 0.031527547610634275])
#
# std_train = np.array([0.3572610932095403, 0.3572878634587769, 0.3572964354695759])
#
# mean_test = np.array([ 0.02296786544189451, 0.022306302832031222, 0.022063635363769547])
#
# std_test = np.array([ 0.24786202451105738, 0.2478912674977951, 0.247902638132196])

mean_train = np.array([0.013921553298525463,-1.1449500105855021e-05,-2.7007798883849245e-05,0.001769272877592398,55.16588351392994,153.9805915845695,164.2891229026677])

std_train = np.array([0.054676044629036594,0.004880067511338395,0.005236856562000297,0.0026666559130332106,28.231218814545844,56.788552659249156,54.81684743513844])

mean_test = np.array([0.016184251541125894,8.372607662649334e-06,-1.7316096806462553e-05,0.0019413151591079851,38.152412915751874,181.7546737559521,135.3366737087278])

std_test = np.array([0.03913930128557607,0.006226349734972811,0.005632704342127379,0.004535597697416357,12.084150617994213,42.01484937597925,38.80465520312229])
# mean_train = np.array([ 0.03654346686172481, 0.035981434774398705, 0.03582025023651128])
#
# std_train = np.array([ 0.3996154020609027, 0.3996487543965463, 0.399658521957082])
#
# mean_test = np.array([0.01873571234566827, 0.018264891034080848, 0.01806365490504674])
#
# std_test = np.array([0.16928838041798588, 0.1692964116284298, 0.1693021845664508])
def get_idx(channels):
    assert channels in [2, 3, 4, 7, 6, 8]
    if channels == 6:
        return list(range(6))
    elif channels == 4:
        return list(range(4))
    elif channels == 7:
        return list(range(7))
    elif channels == 3:
        return list(range(3))
    elif channels == 8:
        return list(range(8))

def getTransform(train=True, channel_idx=[0,1,2,3,4,5,6]):
    if train:
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(), #试试删除与否能否提高精度
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean_train[channel_idx],std=std_train[channel_idx])
            ]
        )
    else:
            transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_test[channel_idx], std=std_test[channel_idx])
        ])
    return transform

_transform_test = _transform.Compose([
            _transform.ToTensor(),
            _transform.Normalize(mean=mean_test, std=std_test)
        ])


class semData(Dataset):
    def __init__(self, train=True, root=r'E:\nxy\net\Data', channels = 7 , transform=None, selftest_dir=None): #modified here!    channels = 35
        self.train = train
        self.root = root
        self.dir = traindir if self.train else testdir
        if selftest_dir is not None:
            self.dir = selftest_dir
        self.channels = channels
        self.c_idx = get_idx(self.channels)
        if selftest_dir is not None: #modified here!
            self.file = os.path.join(self.root,selftest_dir,'labels_0-1')
        else:
            self.file = trainfile if train else testfile

        self.img_dir = os.path.join(self.root, self.dir, imagedir)
        #print(self.img_dir)
        self.label_dir = os.path.join(self.root, self.dir, labeldir)


        if transform is not None:
            self.transform = transform
        else:
            self.transform = getTransform(self.train, self.c_idx)
        
        #self.data_list = pd.read_csv(self.file).values
        #print(self.data_list)
        imges_sets = os.listdir(self.file)
        imges_sets = np.expand_dims(imges_sets, axis=0)
        self.data_list = imges_sets.reshape(-1,1)
        #print(self.data_list)
        print(len(self.data_list))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        L = []
        lbl_name = self.data_list[index][0] # 1081.png
        # p = self.data_list[index].split('.')[0]
        p = lbl_name.split('.')[0]          # 1081
        for k in self.c_idx:
            img_path = p + '.tif'   #'1_'+p+'.tif'
            img_path = os.path.join(self.img_dir, channel_list[k], img_path)
            img =tf.imread(img_path)
            img = np.expand_dims(np.array(img), axis=2)
            L.append(img)

        # image = cv2.imread(os.path.join(self.root,img_path), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        image = np.concatenate(L, axis=-1)
        label = cv2.imread(os.path.join(self.label_dir, lbl_name), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + p + " " + lbl_name + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        return {
            'X':image,
            'Y':label,
            'path': lbl_name       
        }
    
    def TestSetLoader(self,root=r'D:\BARELAND\NET\Data\test',file='test'):
        l = pd.read_csv(os.path.join(root,file)).values
        for i in l:
            filename = i[0]
            path = os.path.join(root, filename)
            image = Image.open(path)
            image = _transform_test(image)
            yield filename,image

if __name__ == "__main__":
    trainset = semData(train=False, channels=5, transform=None, selftest_dir='test')
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        img = data['X']
        label = data['Y']
        path = data['path']
        print(img.size(),label.max())

    

