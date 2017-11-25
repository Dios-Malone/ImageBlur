import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, pause
from scipy.signal import gaussian, convolve2d
from numpy.fft import fft2, ifft2

ker = np.eye(5)/5

def wiener_filter(img, kernel, K = .1):
    temp_img = np.copy(img)
    kernel = np.pad(kernel, [(0, temp_img.shape[0] - kernel.shape[0]), (0, temp_img.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    temp_img = fft2(temp_img)
    kernel = fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    temp_img = kernel * temp_img
   # temp_img = np.conj(temp_img) / np.conj(kernel)
    temp_img = np.real(ifft2(temp_img))
    return temp_img

im = imageio.imread("test.bmp")
im = np.asarray(im)

outIm = wiener_filter(im, ker)

imshow(outIm, 'gray')
plt.show()
imageio.imwrite("output.bmp", outIm) 
print 'done'