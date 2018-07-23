import tensorflow as tf
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(input_shape, n_filters, filter_sizes,z_dim, x, var_G, reuse=False):
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
                var_G.append(W)
            else:
                W = var_G[idx]
                idx+=1
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.elu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        W_fc1 = tf.Variable(tf.random_normal([z_flat_dim, z_dim]))
        var_G.append(W_fc1)
        z = tf.matmul(z_flat,W_fc1)
        #z =  tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        #z = tf.nn.elu(z)
        z_value = z
        W_fc2 = tf.Variable(tf.random_normal([z_dim, z_flat_dim]))
        var_G.append(W_fc2)
        z_ = tf.matmul(z,W_fc2)
        #z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        #z_ = tf.nn.elu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])        
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            if reuse == False:
                W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
                var_G.append(W)
            else:
                W = var_G[idx]
                idx+=1
            decoder.append(W)
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            #if layer_i == len(n_filters)-2:
            #    output = tf.nn.sigmoid(deconv)
            #else:
            deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        
        encoder.reverse()
        shapes_enc.reverse()
        decoder.reverse()
        shapes_dec.reverse()
        
    with tf.name_scope("Decoder"):
        for layer_i, shape in enumerate(shapes_dec):
            W_dec = decoder[layer_i]
            conv = tf.nn.conv2d(current_input, W_dec, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.elu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        z = tf.matmul(tf.layers.flatten(current_input), tf.transpose(W_fc2))
        #z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        #z = tf.nn.elu(z)
        z_transpose = z
        z_ = tf.matmul(z, tf.transpose(W_fc1))
        #z_ =  tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        #z_ = tf.nn.elu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])         
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            #if layer_i == len(n_filters)-2:
            #    output = tf.nn.sigmoid(deconv)
            #else:
            deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.relu(deconv)
            current_input = output
        a = current_input      

    return g, a, z_value, z_transpose

def discriminator(input_shape, n_filters, filter_sizes, x, var_D, reuse=False):
    current_input = x    
    idx = 0
    with tf.name_scope("Discriminator"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            if reuse ==False:
                W = tf.Variable(xavier_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
                var_D.append(W)
            else:
                W = var_D[idx]
                idx+=1
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.elu(conv)
            current_input = output

        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        if reuse ==False:
            W = tf.Variable(tf.random_normal([z_flat_dim,1]))
            b = tf.Variable(tf.zeros(shape=[1]))
            var_D.append(W)
            var_D.append(b)
        else:
            W = var_D[idx]
            idx+=1
            b = var_D[idx]
            idx+=1
        d = tf.nn.xw_plus_b(z_flat,W,b)
    return d

def gradient_penalty(G_sample, A_true_flat, mb_size,input_shape, n_filters, filter_sizes, var_D, reuse=True):
    epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
    X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
    D_X_hat = discriminator(input_shape, n_filters, filter_sizes, X_hat,var_D, reuse)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = list(range(1, X_hat.shape.ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    return gradient_penalty 

