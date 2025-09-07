import numpy as np
from PIL import Image
from scipy import signal
from matplotlib import pyplot as plt


img=Image.open('LenaRGB.tif').convert('L')
img_arr=np.array(img,dtype=np.float64)
l,w=np.shape(img_arr)
plt.subplot(2,3,1)
plt.title("original image")
plt.imshow(img,cmap='gray')
plt.axis('off')
#高斯滤波
b=np.array([[1,2,1],
            [2,4,2],
            [1,2,1]])
b=1/16*b
A=signal.convolve2d(img_arr,b,boundary='symm',mode='same')
plt.subplot(2,3,2)
plt.title('Gas image')
plt.imshow(A,cmap='gray')
plt.axis('off')
#plt.show()

#计算梯度幅值和角度
dx=np.array([[-1,0,1],
            [-1,0,1],
             [-1,0,1]])
dy=np.array([[1,1,1],
             [0,0,0],
             [-1,-1,-1]])
img_x=signal.convolve2d(A,dx,boundary='symm',mode='same')
img_y=signal.convolve2d(A,dy,boundary='symm',mode='same')
img_xy=np.sqrt(img_x**2+img_y**2)
a=np.arctan(img_y,img_x)
a=np.degrees(a)

plt.subplot(2,3,3)
plt.title('gradient image')
plt.imshow(img_xy,cmap='gray')
plt.axis('off')
#plt.show()

plt.subplot(2,3,4)
plt.title('angle image')
plt.imshow(a,cmap='binary')
plt.axis('off')
#plt.show()

#非极大值抑制
#归一化方向
for i in range(l):
    for j in range(w):
        if ((a[i, j] >= -22.5) and (a[i, j] < 0) or (a[i, j] >= 0) and (a[i, j] < 22.5) or (a[i, j] <= -157.5) and (a[i, j] >= -180) or (a[i, j] >= 157.5) and (a[i, j] <= 180)):
            a[i,j] = 0
        elif((a[i, j] >= 22.5) and (a[i, j] < 67.5) or (a[i, j]<= -112.5) and (a[i, j] > -157.5)):
            a[i, j] = -45
        elif((a[i, j] >= 67.5) and (a[i, j] < 112.5) or (a[i, j] <= -67.5) and (a[i, j] > - 112.5)):
            a[i, j] = 90
        elif((a[i, j] >= 112.5) and (a[i, j] < 157.5) or (a[i, j] <= -22.5) and (a[i, j] > -67.5)):
            a[i, j] = 45
n=np.zeros((l,w))
for i in range(1,l-1):
    for j in range(1,w-1):
        if a[i,j]==0 and img_xy[i,j]==max(img_xy[i-1,j],img_xy[i,j],img_xy[i+1,j]):
            n[i,j]=img_xy[i,j]
        elif a[i,j]==-45 and img_xy[i,j]==max(img_xy[i-1,j-1],img_xy[i,j],img_xy[i+1,j+1]):
            n[i,j]=img_xy[i,j]
        elif a[i,j]==45 and img_xy[i,j]==max(img_xy[i-1,j+1],img_xy[i,j],img_xy[i+1,j-1]):
            n[i,j]=img_xy[i,j]
        elif a[i,j]==90 and img_xy[i,j]==max(img_xy[i,j-1],img_xy[i,j],img_xy[i,j+1]):
            n[i,j]=img_xy[i,j]

plt.subplot(2,3,5)
plt.title('NMS image')
plt.imshow(n,cmap='gray')
plt.axis('off')
#plt.show()

#双阈值检测和边缘连接
s=np.zeros((l,w))
tb=0.17*np.max(n)
ts=0.2*np.max(n)
for i in range(l):
    for j in range(w):
        if n[i,j]>=tb:
            s[i,j]=1
        elif n[i,j]<ts:
            s[i,j]=0
        elif n[i-1,j-1]>=tb or n[i-1,j]>=tb or n[i-1,j+1]>=tb or n[i,j-1]>=tb or n[i,j+1]>=tb or n[i+1,j-1]>=tb or n[i+1,j]>=tb or n[i+1,j+1]>=tb:
            s[i,j]=1
plt.subplot(2,3,6)
plt.title('canny image')
plt.imshow(s,cmap='gray')
plt.axis('off')
plt.show()







