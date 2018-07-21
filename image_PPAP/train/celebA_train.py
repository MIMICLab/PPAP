import tensorflow as tf
import numpy as np
import os
import math
import time

from utils import *
from ppap_wgan import *

from glob import glob
from random import shuffle
from download import download_celeb_a

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec


def plot(samples):
    fig = plt.figure(figsize=(32,5))
    gs = gridspec.GridSpec(5,32)
    gs.update(wspace=0.05, hspace=0.05)
        
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64,3), cmap='Greys_r')

    return fig
mb_size = 128
X_dim = 4096
download_celeb_a("../data")
data_files = glob(os.path.join("../data/celebA/*.jpg"))
len_x_train = len(data_files)
sample = [get_image(sample_file, 108, True, 64, is_grayscale = 0) for sample_file in data_files]
sample_images = np.array(sample).astype(np.float32)  
x_train = sample_images

x_train = normalize(x_train)
    
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        #input placeholder
        X = tf.placeholder(tf.float32, shape=[None,64,64,3])
        A_true_flat = X 
        z_dim = 256     
        #autoencoder variables
        var_G = []
        var_A = []
        input_shape=[None, 64, 64, 3]
        n_filters=[3, 128, 256, 512, 1024]
        filter_sizes=[5, 5, 5, 5, 5]
        
        #discriminator variables
        WX1 = tf.Variable(xavier_init([5,5,3,128]))
        WX2 = tf.Variable(xavier_init([5,5,128,256]))
        WX3 = tf.Variable(xavier_init([5,5,256,512]))
        WX4 = tf.Variable(xavier_init([5,5,512,1024]))
        WX5 = tf.Variable(xavier_init([4*4*1024, 1]))
        bX5 = tf.Variable(tf.zeros(shape=[1]))
        var_DX = [WX1,WX2,WX3,WX4,WX5,bX5] 
        
        WZ = tf.Variable(xavier_init([z_dim,1]))
        bZ = tf.Variable(tf.zeros(shape=[1]))
        var_DZ = [WZ,bZ]
        
        G_sample, A_sample = generator(input_shape, n_filters, filter_sizes,3, A_true_flat, var_G)
        re_true, z_true, DZ_real_logits  = Z_discriminator(input_shape, n_filters, filter_sizes,3,z_dim, A_true_flat, var_A,var_DZ)
        re_fake, z_fake, DZ_fake_logits  = Z_discriminator(input_shape, n_filters, filter_sizes,3,z_dim, G_sample, var_A, var_DZ, reuse = True)
        
        DX_real_logits = X_discriminator(A_true_flat, var_DX)
        DX_fake_logits = X_discriminator(G_sample, var_DX)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        dp_epsilon = 0.1
        #A_loss = laploss(A_true_flat,A_sample)
        A_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample,2))
        Z_loss = tf.reduce_mean(tf.pow(z_true - z_fake,2))
        opt_loss = 10.0*A_loss -10.0*Z_loss
        #A_D_loss = laploss(A_true_flat, re_true)
        A_D_loss = tf.reduce_mean(tf.pow(A_true_flat - re_true,2))
        gp_x = X_gradient_penalty(G_sample,A_true_flat, var_DX, mb_size)
        gp_z = Z_gradient_penalty(G_sample,A_true_flat, mb_size,input_shape, n_filters, filter_sizes,3,z_dim, G_sample, var_A, var_DZ, reuse = True)
        
        D_Z_loss = tf.reduce_mean(DZ_fake_logits) - tf.reduce_mean(DZ_real_logits) + 10.0*gp_x
        D_X_loss = tf.reduce_mean(DX_fake_logits) - tf.reduce_mean(DX_real_logits) + 10.0*gp_z
        
        D_penalty = tf.abs(tf.abs(D_X_loss - D_Z_loss) - dp_epsilon)
        
        G_loss = -tf.reduce_mean(DZ_fake_logits) - tf.reduce_mean(DX_fake_logits) + opt_loss
        D_loss = D_Z_loss + D_X_loss + D_penalty + 0.1*A_D_loss
        
        tf.summary.image('Original',A_true_flat)
        tf.summary.image('G_sample',G_sample)
        tf.summary.image('A_sample',A_sample)
        tf.summary.image('D_sample',re_true)
        tf.summary.scalar('D_Z_loss', D_Z_loss)
        tf.summary.scalar('D_X_loss', D_X_loss)
        tf.summary.scalar('D_Penalty',D_penalty)
        tf.summary.scalar('G_loss',- tf.reduce_mean(DZ_fake_logits) - tf.reduce_mean(DX_fake_logits))   
        tf.summary.scalar('A_G_opt_loss',A_loss)
        tf.summary.scalar('A_D_opt_loss',A_D_loss)
        tf.summary.scalar('Z_opt_loss',Z_loss)
        merged = tf.summary.merge_all()

        num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
       
        D_solver = tf.train.AdamOptimizer(learning_rate=5e-5,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=var_DX+var_A+var_DZ, global_step=global_step)
        G_solver = tf.train.AdamOptimizer(learning_rate=5e-5,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)
        
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "../results/models/celebA" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists('../results/models/'):
            os.makedirs('../results/models/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists('../results/dc_out_celebA/'):
            os.makedirs('../results/dc_out_celebA/')
        #if not os.path.exists('../results/generated_mnist/'):
        #    os.makedirs('../results/generated_mnist/')            

        train_writer = tf.summary.FileWriter('../results/graphs/'+'celebA',sess.graph)
        sess.run(tf.global_variables_initializer())
        i = 0       
        for it in range(1000000000):
            for _ in range(5):
                X_mb = next_batch(mb_size, x_train)
                _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb})
            summary, _, G_loss_curr = sess.run([merged,G_solver, G_loss],feed_dict={X: X_mb})
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary,current_step)
        
            if it % 100 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4};'.format(it,D_loss_curr, G_loss_curr))

            if it % 1000 == 0: 
                samples = sess.run(G_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1, 64, 64, 3]).eval()
                img_set = np.append(X_mb[:32], samples_flat[:32], axis=0)
                
                samples = sess.run(A_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1, 64,64, 3]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0) 

                samples = sess.run(re_true, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1,64, 64, 3]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0) 
                
                samples = sess.run(re_fake, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1, 64, 64, 3]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0)
                
                fig = plot(img_set)
                plt.savefig('../results/dc_out_celebA/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model at {} at step {}'.format(path, current_step))
'''
            if it% 100000 == 0 and it != 0:
                for ii in range(len_x_train//100):
                    xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
                    samples = sess.run(G_sample, feed_dict={X: xt_mb})
                    if ii == 0:
                        generated = samples
                    else:
                        np.append(generated,samples,axis=0)
                np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)

    for iii in range(len_x_train//100):
        xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
        samples = sess.run(G_sample, feed_dict={X: xt_mb})
        if iii == 0:
            generated = samples
        else:
            np.append(generated,samples,axis=0)
    np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)
'''