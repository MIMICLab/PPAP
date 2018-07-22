import tensorflow as tf
import numpy as np
from utils.utils import *
from model import *
import sys
import os
import math
import time
from utils.data_helper import data_loader

dataset = sys.argv[1]

mb_size, X_dim, width, height, channels,len_x_train, x_train = data_loader(dataset)
    
    
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        #input placeholder
        input_shape=[None, width, height, channels]
        hidden = 128
        n_filters=[channels, hidden, hidden*2, hidden*3, hidden*4]
        filter_sizes=[5, 5, 5, 5, 5]
  
        X = tf.placeholder(tf.float32, shape=[None, width, height,channels])
        A_true_flat = X
        
        #autoencoder variables
        var_G = [] 
        var_D = []
        var_H = []
        
        G_sample, A_sample = generator(input_shape, n_filters, filter_sizes,A_true_flat, var_G)
        D_real_logits = discriminator(input_shape, n_filters, filter_sizes, A_true_flat, var_D)
        D_fake_logits = discriminator(input_shape, n_filters, filter_sizes, G_sample, var_D, reuse = True)
        decoded_fake, encoded_true = hacker(input_shape, n_filters, filter_sizes, G_sample, A_true_flat, var_H)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        A_G_loss = laploss(A_true_flat,A_sample)
        A_D_true_loss = laploss(G_sample, encoded_true)
        A_D_fake_loss = laploss(A_true_flat, decoded_fake)   
        
        gp = gradient_penalty(G_sample, A_true_flat, mb_size,input_shape, n_filters, filter_sizes, var_D, reuse=True)
        D_loss = tf.reduce_mean(D_fake_logits)-tf.reduce_mean(D_real_logits) +10.0*gp
        
        H_loss = 0.01*A_D_true_loss + 0.01*A_D_fake_loss
        D_total_loss = D_loss + H_loss
        G_loss = -tf.reduce_mean(D_fake_logits) + 0.01*A_G_loss - H_loss
        
        tf.summary.image('Original',A_true_flat)
        tf.summary.image('G_sample',G_sample)
        tf.summary.image('A_sample',A_sample)
        tf.summary.image('encoded_true', encoded_true)
        tf.summary.image('decoded_fake',decoded_fake)
        tf.summary.scalar('D_loss', D_loss)      
        tf.summary.scalar('G_loss',-tf.reduce_mean(D_fake_logits))   
        tf.summary.scalar('A_G_loss',A_G_loss)
        tf.summary.scalar('A_D_true_loss',A_D_true_loss)
        tf.summary.scalar('A_D_fake_loss',A_D_fake_loss)
        merged = tf.summary.merge_all()

        num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
       
        D_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_total_loss,var_list=var_D+var_H, global_step=global_step)
        G_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)

        timestamp = str(int(time.time()))
        if not os.path.exists('results/'):
            os.makedirs('results/')        
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "results/models/{}_".format(dataset) + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists('results/models/'):
            os.makedirs('results/models/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists('results/dc_out_{}/'.format(dataset)):
            os.makedirs('results/dc_out_{}/'.format(dataset))
        if not os.path.exists('results/generated_{}/'.format(dataset)):
            os.makedirs('results/generated_{}/'.format(dataset))            

        train_writer = tf.summary.FileWriter('results/graphs/{}'.format(dataset),sess.graph)
        sess.run(tf.global_variables_initializer())
        i = 0       
        for it in range(1000000000):
            for _ in range(5):
                if dataset == 'mnist':
                    X_mb, _ = x_train.train.next_batch(mb_size)
                    X_mb = np.reshape(X_mb,[-1,28,28,1])
                else:
                    X_mb = next_batch(mb_size, x_train)
                _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb})
            summary, _, G_loss_curr = sess.run([merged,G_solver, G_loss],feed_dict={X: X_mb})
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary,current_step)
        
            if it % 100 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4};'.format(it,D_loss_curr, G_loss_curr))

            if it % 1000 == 0: 
                G_sample_curr,A_sample_curr,re_true_curr,re_fake_curr = sess.run([G_sample,A_sample,decoded_fake, encoded_true], feed_dict={X: X_mb})
                samples_flat = tf.reshape(G_sample_curr,[-1,width,height,channels]).eval()
                img_set = np.append(X_mb[:32], samples_flat[:32], axis=0)
                samples_flat = tf.reshape(A_sample_curr,[-1,width,height,channels]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0) 
                samples_flat = tf.reshape(re_true_curr,[-1,width,height,channels]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0) 
                samples_flat = tf.reshape(re_fake_curr,[-1,width,height,channels]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0)
                
                fig = plot(img_set, width, height, channels)
                plt.savefig('results/dc_out_{}/{}.png'.format(dataset,str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model at {} at step {}'.format(path, current_step))

            if it% 100000 == 0 and it!=0:
                for ii in range(len_x_train//100):
                    if dataset == 'mnist':
                        Xt_mb, _ = x_train.train.next_batch(100)
                        Xt_mb = np.reshape(Xt_mb,[-1,28,28,1])
                    else:
                        Xt_mb = next_batch(100, x_train)
                    samples = sess.run(G_sample, feed_dict={X: Xt_mb})
                    if ii == 0:
                        generated = samples
                    else:
                        generated = np.concatenate((generated,samples), axis=0)
                np.save('results/generated_{}/generated_image.npy'.format(dataset), generated)

    for iii in range(len_x_train//100):
        xt_mb, _ = mnist.train.next_batch(100,shuffle=False)
        samples = sess.run(G_sample, feed_dict={X: xt_mb})
        if iii == 0:
            generated = samples
        else:
            generated = np.concatenate((generated,samples), axis=0)
    np.save('results/generated_{}/generated_image.npy'.format(dataset), generated)
