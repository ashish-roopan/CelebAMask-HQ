import os
import cv2
import argparse
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

def make_folder(path):
  if not os.path.exists(os.path.join(path)):
    print(os.path.join(path))
    os.makedirs(os.path.join(path))
    
def add_parts(mask,parts,base_color_mask=None,base_mask =None):
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    color_list = [ [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    
    if base_color_mask is not None:
        color_mask=base_color_mask
        mask_white=base_mask
    else:
        color_mask=np.zeros((512,512,3),dtype=np.uint8)
        mask_white=np.zeros((512,512,3),dtype=np.uint8)
    for part in parts:
        color_mask[mask==(label_list.index(part)+1)]=color_list[label_list.index(part)]
        mask_white[mask==(label_list.index(part)+1)]=label_list.index(part)
        
    return color_mask,mask_white






image_path= '/content/CelebAMask-HQ/pipeline/data/mask'
images=os.listdir(image_path)
images=np.sort(images)
save_path='/content/CelebAMask-HQ/pipeline/data/mask_m/'
make_folder(save_path)

head_parts=['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r']
body_parts=['neck_l', 'neck', 'cloth']

####read images and mask
for i in range(len(images)//2):
  src_mask= np.array(Image.open(os.path.join(image_path,images[i+len(images)//2])))
  dst_mask= np.array(Image.open(os.path.join(image_path,images[i])))  
  
  dst_body_color,dst_body_mask=add_parts(dst_mask,body_parts)
  color_mask,mask=add_parts(src_mask,head_parts,dst_body_color,dst_body_mask)
  color_mask= cv2.cvtColor(color_mask,cv2.COLOR_BGR2RGB)
  cv2.imwrite(save_path+images[i][3:],color_mask)
  print(images[i][3:])
  # cv2_imshow(color_mask)
  