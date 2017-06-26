import os
import glob
from fnmatch import fnmatch
from scipy import  misc
import  numpy as np

DATA_MAIN_PATH = '/media/ngocminh/DATA/EURECOM_Kinect_Face_Dataset'
RGB_4D_SAVE_PATH = '/media/ngocminh/DATA/EURECOM_Kinect_Face_Dataset_4D_BIN'
depth_bmp  = []

for path, subdirs, files in os.walk(DATA_MAIN_PATH):
    for name in files:
        if fnmatch(name, 'depth_*.bmp'):
            depth_bmp.append(os.path.join(path, name))

rgb_bmp = []
for path, subdirs, files in os.walk(DATA_MAIN_PATH):
    for name in files:
        if fnmatch(name, 'rgb*.bmp'):
            rgb_bmp.append(os.path.join(path, name))

if not os.path.exists(RGB_4D_SAVE_PATH):
    os.mkdir(RGB_4D_SAVE_PATH)


file_lists = []
for depth,rgb in zip(depth_bmp,rgb_bmp):
    saveName = depth.split('/')[-1].replace('depth','4drgb')
    saveName = saveName.replace('bmp', 'bin')
    print saveName

    saveFolder = depth.replace(DATA_MAIN_PATH, './')
    saveFolder = '/'.join(saveFolder.split('/')[1:4])[1:]
    fullsaveFolder = os.path.join(RGB_4D_SAVE_PATH, saveFolder)
    print saveFolder, fullsaveFolder

    file_lists.append(saveFolder+'/'+saveName)
    ''''
    depth_img = misc.imread(depth)
    rgb_img = misc.imread(rgb)

    print depth_img.shape
    print rgb_img.shape

    img4d = np.dstack((rgb_img,depth_img))

    if not os.path.exists(fullsaveFolder):
        os.makedirs(fullsaveFolder)

    img4d.tofile(fullsaveFolder+'/'+saveName)
    '''