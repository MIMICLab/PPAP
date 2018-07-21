import tensorflow as tf
import numpy as np

initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def generator(input_shape, n_filters, filter_sizes, last_layer, x, theta_G, reuse=False):
    current_input = x    
    encoder = []
    decoder = []
    shapes_enc = []
    shapes_dec = []
    idx = 0
    with tf.name_scope("G_Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            if reuse ==False:
                W = tf.Variable(xavier_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
                theta_G.append(W)
            else:
                W = theta_G[idx]
                idx+=1
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            if reuse == False:
                W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
                theta_G.append(W)
            else:
                W = theta_G[idx]
                idx+=1
            decoder.append(W)
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == last_layer:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        
        encoder.reverse()
        shapes_enc.reverse()
        decoder.reverse()
        shapes_dec.reverse()
        
    with tf.name_scope("G_Decoder"):
        for layer_i, shape in enumerate(shapes_dec):
            W_dec = decoder[layer_i]
            conv = tf.nn.conv2d(current_input, W_dec, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()         
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == last_layer:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input      

    return g, a

def Z_discriminator(input_shape, n_filters, filter_sizes, last_layer, z_dim, x, theta_A, theta_DZ, reuse=False):
    current_input = x    
    encoder = []
    decoder = []
    shapes_enc = []
    shapes_dec = []
    idx = 0
    with tf.name_scope("Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            if reuse ==False:
                W = tf.Variable(xavier_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
                theta_A.append(W)
            else:
                W = theta_A[idx]
                idx+=1
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        if reuse == False:
            W_fc = tf.Variable(tf.random_normal([4*4*n_filters[-1], z_dim]))
            theta_A.append(W_fc)
        else:
            W_fc = theta_A[idx]
            idx+=1
        z = tf.matmul(tf.layers.flatten(current_input),W_fc)
        z =  tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z = tf.nn.tanh(z)
        z_value = z
        z_ = tf.matmul(z,tf.transpose(W_fc))
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])
        for layer_i, shape in enumerate(shapes_enc):
            W = encoder[layer_i]
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == last_layer:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        h = z_value
        d = tf.nn.xw_plus_b(h,theta_DZ[0],theta_DZ[1])
        return g, z_value, d
    
def X_discriminator(x,var_D):
    current_input = x
    with tf.name_scope("X_Discriminator"):
        for i in range(len(var_D)-2):
            conv = tf.nn.conv2d(current_input, var_D[i], strides=[1,2,2,1],padding='SAME')
            conv = tf.contrib.layers.layer_norm(conv)
            current_input = tf.nn.leaky_relu(conv)            
        h = tf.layers.flatten(current_input)     
        d = tf.nn.xw_plus_b(h, var_D[-2], var_D[-1])        
    return d

def z_pyramid_loss(G_pyramid, A_pyramid, original):
    z_losses = [tf.norm(g-a,ord='euclidean')/tf.cast(tf.size(g, out_type=tf.int32),tf.float32) for g,a in zip(G_pyramid,  A_pyramid)]
    z_loss = tf.reduce_sum(z_losses)*tf.cast(tf.shape(original, out_type=tf.int32),tf.float32)[0]
    return  z_loss

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

def X_gradient_penalty(G_sample, A_true_flat, var_D, mb_size):
    epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
    X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
    D_X_hat = X_discriminator(X_hat,var_D)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = list(range(1, X_hat.shape.ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    return gradient_penalty

def Z_gradient_penalty(G_sample, A_true_flat, mb_size,input_shape, n_filters, filter_sizes, last_layer, z_dim, x, theta_A, theta_DZ, reuse=True):
    epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
    X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
    D_X_hat = Z_discriminator(input_shape, n_filters, filter_sizes,last_layer,z_dim, X_hat, theta_A, theta_DZ, reuse)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = list(range(1, X_hat.shape.ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    return gradient_penalty