import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


import cv2 
from PIL import Image 
import os
from os import listdir 
from os.path import isfile, join 

def load_images_from_folder(folder):
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs, ret = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w = int(28)
            h = int(28)
            maxi = 0
            
            for c in cnt:
                x,y,w,h = cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            
            im_crop = thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop, (28,28))
            im_resize = np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    
    return train_data




data=load_images_from_folder('-')

for i in range(0, len(data)):
    data[i]=np.append(data[i],['10'])

print(len(data))

data11=load_images_from_folder('+')

for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['11'])
    
data=np.concatenate((data,data11))
print(len(data))

       
data0=load_images_from_folder('0')
for i in range(0,len(data0)):
    data0[i]=np.append(data0[i],['0'])
data=np.concatenate((data,data0))
print(len(data))

data1=load_images_from_folder('1')

for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data=np.concatenate((data,data1))
print(len(data))

data2=load_images_from_folder('2')

for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data=np.concatenate((data,data2))
print(len(data))

data3=load_images_from_folder('3')

for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data=np.concatenate((data,data3))
print(len(data))

data4=load_images_from_folder('4')

for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data=np.concatenate((data,data4))
print(len(data))

data5=load_images_from_folder('5')

for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data=np.concatenate((data,data5))
print(len(data))

data6=load_images_from_folder('6')

for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data=np.concatenate((data,data6))
print(len(data))

data7=load_images_from_folder('7')

for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data=np.concatenate((data,data7))
print(len(data))

data8=load_images_from_folder('8')

for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data=np.concatenate((data,data8))
print(len(data))


data9=load_images_from_folder('9')

for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data=np.concatenate((data,data9))
print(len(data))

data12=load_images_from_folder('times')

for i in range(0,len(data12)):
    data12[i]=np.append(data12[i],['12'])
data=np.concatenate((data,data12))
print(len(data))

df=pd.DataFrame(data,index=None)
df.to_csv('train.csv',index=False)
