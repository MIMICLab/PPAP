from random import shuffle
import scipy.misc
import numpy as np
import tensorflow as tf
import cv2
from scipy.misc import toimage
import scipy.io as sio
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec

def next_batch(num, data, shuffle=True):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    if shuffle == True:
        np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def plot(samples, width, height, channels):
    fig = plt.figure(figsize=(32,4))
    gs = gridspec.GridSpec(4,32)
    gs.update(wspace=0.05, hspace=0.05)
    norm=plt.Normalize(0, 1)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channels == 1:
            plt.imshow(norm(sample.reshape(width, height)), cmap='Greys_r')
        else:
            plt.imshow(norm(sample.reshape(width, height,channels)))           
    return fig

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

# def merge(images, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[j*h:j*h+h, i*w:i*w+w, :] = image
#     return img

#modified pyramid Lap loss from https://github.com/mtyka/laploss/blob/master/laploss.py
def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),[k_size, k_size, 1, 1])
    t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
    t_result = t_input
    for r in range(repeats):
        t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3, strides=[1, stride, stride, 1], padding='SAME')
    return t_result

def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
        t_pyr.append(current)
    return t_pyr

def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = [tf.norm(a-b,ord=1)/tf.cast(tf.size(a, out_type=tf.int32),tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
    t_loss = tf.reduce_sum(t_losses)*tf.cast(tf.shape(t_img1, out_type=tf.int32),tf.float32)[0]
    return t_loss

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/255.

# def inverse_transform(images):
#     return (images+1.)/2.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

# def imsave(images, size, path):
#     return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

# def save_images(images, size, image_path):
#     return imsave(inverse_transform(images), size, image_path)

def add_noise_to_gradients(grads_and_vars, epsilon, sparse_grads=False):
    if not isinstance(grads_and_vars, list):
        raise ValueError('`grads_and_vars` must be a list.')
    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, tf.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
            noise = np.random.laplace(0.0,epsilon,gradient_shape)
            noisy_gradients.append(gradient + noise)
    return zip(noisy_gradients,variables)
