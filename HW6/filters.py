import cv2
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


def custom_filter(image, template):
    if not (template.shape[0] == template.shape[1]):
        raise ValueError()
    if template.shape[0] % 2 == 0:
        raise ValueError()
    filtersize05 = template.shape[0] // 2
    
    filtered_image = np.zeros_like(image)

    s = template.sum()
    window = np.array([coef / s for coef in template.flatten()])
    shape = filtered_image.shape
    wid = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            for color in range(shape[2]):
                wid = 0
                for m in range(i - filtersize05, i + filtersize05 + 1):
                    for n in range(j - filtersize05, j + filtersize05 + 1):
                        if 0 <= m and m < shape[0] and 0 <= n and n < shape[1]:
                            filtered_image[i][j][color] += window[wid] * image[m][n][color]
                        wid += 1

    return filtered_image


def custom_filter_gpu(image, template):
    if not (template.shape[0] == template.shape[1]):
        raise ValueError()
    if template.shape[0] % 2 == 0:
        raise ValueError()
    filtersize05 = template.shape[0] // 2
    
    image_gpu = gpuarray.to_gpu(image)
 
    filtered_image = gpuarray.zeros_like(image_gpu)

    s = template.sum()
    window = gpuarray.to_gpu(np.array([coef / s for coef in template.flatten()]))
    shape = filtered_image.shape
    wid = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            for color in range(shape[2]):
                wid = 0
                for m in range(i - filtersize05, i + filtersize05 + 1):
                    for n in range(j - filtersize05, j + filtersize05 + 1):
                        if 0 <= m and m < shape[0] and 0 <= n and n < shape[1]:
                            filtered_image[i][j][color] += window[wid] * image_gpu[m][n][color]
                        wid += 1

    return filtered_image.get()


def median_filter(image, filter_size=5):
    if filter_size % 2 == 0:
        raise ValueError()
    filtersize05 = filter_size // 2

    filtered_image = image.copy()

    window = np.zeros(filter_size * filter_size, dtype=image.dtype)
    shape = filtered_image.shape
    wid = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            for color in range(shape[2]):
                window.fill(0)
                wid = 0
                for m in range(i - filtersize05, i + filtersize05 + 1):
                    for n in range(j - filtersize05, j + filtersize05 + 1):
                        if 0 <= m and m < shape[0] and 0 <= n and n < shape[1]:
                            window[wid] = image[m][n][color]
                        wid += 1
                window.sort()
                filtered_image[i][j][color] = window[(window.size - 1) // 2]

    return filtered_image


def median_filter_gpu(image, filter_size=5):
    if filter_size % 2 == 0:
        raise ValueError()
    filtersize05 = filter_size // 2

    image_gpu = gpuarray.to_gpu(image)
    filtered_image = image_gpu.copy()
    
    window = gpuarray.zeros(filter_size * filter_size, dtype=image_gpu.dtype)
    shape = np.array(image.shape)
    wid = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            for color in range(shape[2]):
                window.fill(0)
                wid = 0
                for m in range(i - filtersize05, i + filtersize05 + 1):
                    for n in range(j - filtersize05, j + filtersize05 + 1):
                        if 0 <= m and m < shape[0] and 0 <= n and n < shape[1]:
                            window[wid] = image_gpu[m][n][color]
                        wid += 1
                window.set(np.sort(window.get()))
                filtered_image[i][j][color] = window[(window.size - 1) // 2]

    return filtered_image.get()


if __name__ == "__main__":
    image = np.array(cv2.imread("./cat_2.jpg"))
    
    template = np.random.rand(5, 5)
    filtered_image = custom_filter_gpu(image)
    cv2.imwrite("ccat_2.jpg", filtered_image)

    filtered_image = median_filter_gpu(image)
    cv2.imwrite("mcat_2.jpg", filtered_image)
