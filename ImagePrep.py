import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, pause
from scipy.signal import gaussian, convolve2d
from numpy.fft.fftpack import fft2, ifft2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def wiener_filter(img, kernel, K = 10):
    temp_img = np.copy(img)
    kernel = np.pad(kernel, [(0, temp_img.shape[0] - kernel.shape[0]), (0, temp_img.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    temp_img = fft2(temp_img)
    kernel = fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    temp_img = temp_img * kernel
    temp_img = np.abs(ifft2(temp_img))
    return np.uint8(temp_img)

def blur(img, block_size = 3):
    temp_img = np.copy(img)
    h = np.eye(block_size)/block_size
    temp_img = convolve2d(temp_img, h, mode = 'valid')
    return np.uint8(temp_img)

im = imageio.imread("original.png")
im = np.asarray(im)
greyIm = rgb2gray(im)
result = blur(greyIm, 5)
imshow(result, cmap='gray')
plt.show()
imageio.imwrite("test.bmp", result) 

print result.shape
print 'done'