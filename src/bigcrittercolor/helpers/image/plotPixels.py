import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import colorsys
import numpy as np

def plotPixels(img,img_colorspace="rgb", sample_n=2000, centroids=None):
    #convert from BGR to RGB
    if img_colorspace == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #hls2rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lab2rgb = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) / 255

    pixels = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            #if not (pixel[1] == 255):
            #pixel = [p / 255 for p in pixel]
            if img_colorspace == "hls":
                pixels.append(list(colorsys.hls_to_rgb(pixel[0]/360, pixel[1]/255, pixel[2]/255)))
            elif img_colorspace == "lab":
                pixels.append(lab2rgb[i][j])
            else:
                pixels.append(img[i][j] / 255)

    #get rgb values from image to 1D array
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    if sample_n is not None:
        indices = np.random.choice(len(r), size=sample_n, replace=False)
        r = r[indices]
        g = g[indices]
        b = b[indices]
        pixels = [pixels[i] for i in indices]
    #plotting
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b, c=pixels,alpha=0.5)

    if centroids is not None:
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],s=600,c='black',marker='x')

    ax.set_xlabel('Red')
    ax.set_ylabel('Blue')
    ax.set_zlabel('Green')
    plt.show()