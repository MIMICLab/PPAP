import tensorflow as tf
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def generator(input_shape, n_filters, filter_sizes, x, theta_G, reuse=False):
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
            if layer_i == 3:
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
            if layer_i == 3:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input      

    return g, a

def discriminator(input_shape, n_filters, filter_sizes, x, theta_A, reuse=False):
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
        z = current_input
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]    
            if reuse == False:
                W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
                theta_A.append(W)
            else:
                W = theta_A[idx]
                idx+=1            
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == 3:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        reconstructed = current_input
        z_flat = tf.layers.flatten(z)
        z_flat_dim = int(z_flat.get_shape()[1])
        if reuse ==False:
            W = tf.Variable(tf.random_normal([z_flat_dim,z_flat_dim]))
            theta_A.append(W)
        else:
            W = theta_A[idx]
            idx+=1
        h = tf.matmul(z_flat,W)
        h = tf.contrib.layers.batch_norm(h,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h = tf.nn.leaky_relu(h)
        if reuse ==False:
            W = tf.Variable(tf.random_normal([z_flat_dim,1]))
            b = tf.Variable(tf.zeros(shape=[1]))
            theta_A.append(W)
            theta_A.append(b)
        else:
            W = theta_A[idx]
            idx+=1
            b = theta_A[idx]
            idx+=1
        d = tf.nn.xw_plus_b(h,W,b)
        return reconstructed, d    