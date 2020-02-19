
# coding: utf-8

# In[1]:

import pydicom
import numpy as np
import dicom
import png
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import shutil
import operator
import warnings
from dicom_contour.contour import *
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib
import glob
from skimage.color import rgb2gray
from skimage import io, color
import sys
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# In[126]:

dicom_dir=''
new_path=""
marge_dir=""
images_dir=""
masks_dir=""


# In[10]:

def get_contour_dict2(contour_file, path, index):  
    if path[-1] != '/': path += '/'
    contour_list = cfile2pixels(contour_file, path, index)
    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        contour_dict[img_id] = contour_arr
    return contour_dict

def get_contour_dict3(contour_file, path, index):  
    if path[-1] != '/': path += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path, index)
    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        a=fill_contour(contour_arr)
        contour_dict[img_id] = fill_contour(a)
    return contour_dict

def get_img_dict(contour_file, path, index):
    if path[-1] != '/': path += '/'
    contour_list = cfile2pixels(contour_file, path, index)
    img_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        img_dict[img_id] =img_arr
    return img_dict

def draw_contour( img, contours):
    image=np.squeeze(img)
    ax.imshow(image,cmap='gray')  
    ax.set_axis_off()
    
    for i, cnt in enumerate(contours):
        cnt = cnt.squeeze(axis=1)
        ax.add_patch(Polygon(cnt, color="r", fill=None, lw=0.5))
        plt.savefig(marge_dir+"/"+X_name[n][39:-4]+".png")
        #plt.savefig(""+'%d_predict_contour.png'%n)
        
        
def dicom2png(source_folder, output_folder):
    list_of_files = os.listdir(source_folder)
    for file in list_of_files:
        try:
            ds = pydicom.dcmread(os.path.join(source_folder,file))
            shape = ds.pixel_array.shape
            image_2d = ds.pixel_array.astype(float)
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
            image_2d_scaled = np.uint8(image_2d_scaled)

            with open(os.path.join(output_folder,file)+'.png' , 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        except:
            pass


# In[128]:

for j in tqdm(range(,)):#range(@,@) file_number
    path =dicom_dir+str(j)
    for file_name in os.listdir(path):
        a=pydicom.dcmread(path+"/"+file_name)
        filename=a.SOPInstanceUID
        os.rename(path+"/"+file_name,path+"/"+filename+'.dcm')
        contour_file = get_contour_file(path+'/')
        img_format='png'
        if not filename == contour_file[0:-4]:
            X=np.zeros((256,256),dtype=np.uint8)
            plt.imsave(new_path + f'/masks/mask_{filename}.{img_format}', X)

    contour_data = pydicom.read_file(path+ '/' + contour_file)            
    get_roi_names(contour_data)
    target_range =len(get_roi_names(contour_data))-1
    dicom2png(path, new_path+"images")
    con=[]
    for i in range(target_range):
        a=get_contour_dict3(contour_file, path, i)
        con.append(a)
        img=[]
    for i in range(target_range):
        a=get_img_dict(contour_file, path, i)
        img.append(a)
    img_dict={}
    for i in range(target_range):
        for k,v in img[(i)].items():
            img_dict[k] = v  
    con_dict={}
    for i in range(target_range):
        for k,v in con[(i)].items():
            if k in con_dict:
                con_dict[k] += v
            else:
                con_dict[k] = v   

    X=list(con_dict.values())
    Z=list(con_dict.keys())

    img_format='png'
    i=0
    for i in range(len(X)):
        c=Z[i]
        plt.imsave(new_path + f'/masks/mask_{c}.{img_format}', X[i])


# # Mearged ContourMask  and  Images
# 

# In[11]:

new_path = ""
marge_path=new_path+"marge"
path_img=new_path+"images"
path_mask=new_path+"masks"

marge_dir=""

for img in glob.glob(os.path.join(path_img,'*.png')):
    image=Image.open(os.path.join(path_img,img))
    img_gray = image.convert("L")
    img_gray.save(os.path.join(path_img,img))
for img in glob.glob(os.path.join(path_mask,'*.png')):
    image=Image.open(os.path.join(path_mask,img))
    img_gray = image.convert("L")
    img_gray.save(os.path.join(path_mask,img))
    
IMG_WIDTH = 256   
IMG_HEIGHT = 256 
IMG_CHANNELS = 1 
X_name = sorted(glob.glob(path_img+'/*.png'))
Y_name = sorted(glob.glob(path_mask+'/*.png'))

num_of_train_imgs = len(X_name)
num_of_mask_imgs = len(Y_name)

X_imgs = np.zeros((num_of_train_imgs, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_imgs = np.zeros((num_of_train_imgs, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for n in range(num_of_train_imgs):
    X_imgs[n] = imread(X_name[n],as_gray=True).reshape(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    Y_imgs[n] = imread(Y_name[n],as_gray=True).reshape(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
for n in tqdm(range(num_of_train_imgs)):
    fig, ax = plt.subplots(figsize=(8, 8))
    ret,thresh = cv2.threshold(Y_imgs[n],127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    draw_contour(X_imgs[n] , contours)


# In[ ]:



