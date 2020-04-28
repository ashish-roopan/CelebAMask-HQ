# %%writefile vid2frame.py
import cv2
import numpy as np
import os

def make_folder(path):
        if not os.path.exists(os.path.join(path)):
                print(os.path.join(path))
                os.makedirs(os.path.join(path))
def video_2_frame(kind,video):
        data_path='./data'
        video_path=os.path.join(data_path,video)
        print(video_path)
        frame_save_path=os.path.join(data_path,'frames')
        make_folder(frame_save_path)
        
        cap = cv2.VideoCapture(video_path)
        i=0
        while(cap.isOpened()):
                ret,frame = cap.read()
                if not ret:
                        break
                cv2.imwrite(os.path.join(frame_save_path, kind +str(i)+'.jpg'),frame)
                # print(os.path.join(frame_save_path, kind +str(i)+'.jpg'))
                i+=1
                # print(i)
                

video_2_frame('dst','dst_face.mp4')
video_2_frame('src','generated.mp4')
print('1.video converted to frames')