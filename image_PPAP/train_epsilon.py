import tensorflow as tf
import numpy as np
from utils.utils import *
from model import *
import sys
import os
import math
import time
from utils.data_helper import data_loader
from model import xavier_init, he_normal_init

dataset = sys.argv[1]
model_name = sys.argv[2]
prev_iter = int(sys.argv[3])
init_epsilon = 1.0

mb_size, X_dim, width, height, channels,len_x_train, x_train, len_x_test, x_test  = data_loader(dataset)
    
    
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        #input placeholder
        input_shape=[None, width, height, channels]
        filter_sizes=[5, 5, 5, 5, 5]        
        hidden = 128
        z_dim = 128          

        if dataset == 'celebA' or dataset == 'lsun':        
            n_filters=[channels, hidden, hidden*2, hidden*4, hidden*8]
        else:      
            n_filters=[channels, hidden, hidden*2, hidden*4]
            
        X = tf.placeholder(tf.float32, shape=[None, width, height,channels])
        Z_S = tf.placeholder(tf.float32, shape=[None,  z_dim])        
        Z_noise = tf.placeholder(tf.float32, shape=[None,  z_dim])
        A_true_flat = X
        
        #autoencoder variables
        var_A = []
        var_G = []
        var_H = []
        #discriminator variables
        W1 = tf.Variable(he_normal_init([5,5,channels, hidden//2]))
        W2 = tf.Variable(he_normal_init([5,5, hidden//2,hidden]))
        W3 = tf.Variable(he_normal_init([5,5,hidden,hidden*2]))
        if dataset == 'celebA' or dataset == 'lsun':
            W4 = tf.Variable(he_normal_init([5,5,hidden*2,hidden*4]))
            W5 = tf.Variable(xavier_init([4*4*hidden*4, 1]))
            b5 = tf.Variable(tf.zeros(shape=[1]))
            var_D = [W1,W2,W3,W4,W5,b5] 
        else:
            W4 = tf.Variable(xavier_init([4*4*hidden*2, 1]))
            b4 = tf.Variable(tf.zeros(shape=[1]))
            var_D = [W1,W2,W3,W4,b4] 
        
        global_step = tf.Variable(0, name="global_step", trainable=False)        

        G_sample, latent_z, z_noised, epsilon_layer, z_noise,e_var = edp_autoencoder(input_shape, n_filters, filter_sizes,z_dim, A_true_flat, Z_noise, var_A, var_G, init_epsilon, Z_S)
        G_hacked = hacker(input_shape, n_filters, filter_sizes,z_dim, G_sample, var_H)
             
        D_real_logits = discriminator(A_true_flat, var_D)
        D_fake_logits = discriminator(G_sample, var_D)
        
        gp = gradient_penalty(G_sample, A_true_flat, mb_size,var_D)
        dp_epsilon = tf.reduce_mean(epsilon_layer)
        D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) +10.0*gp 

        privacy_gain = tf.reduce_mean(tf.pow(A_true_flat - G_hacked,2))    
        A_loss = tf.reduce_mean(tf.pow(A_true_flat - G_sample,2))  
        G_loss = -tf.reduce_mean(D_fake_logits) - privacy_gain
        H_loss = privacy_gain 
        latent_max = tf.reduce_max(latent_z, axis = 0)
        latent_min = tf.reduce_min(latent_z, axis = 0)
        
        tf.summary.image('Original',A_true_flat)       
        tf.summary.image('fake',G_sample)
        tf.summary.image('decoded_from_fake',G_hacked)
        tf.summary.scalar('D_loss', D_loss)      
        tf.summary.scalar('G_loss',-tf.reduce_mean(D_fake_logits))
        tf.summary.scalar('Encoder_loss', A_loss)
        tf.summary.scalar('privacy_gain',privacy_gain)
        tf.summary.scalar('epsilon', dp_epsilon)
        tf.summary.histogram('epsilon_layer',epsilon_layer)
        tf.summary.histogram('z_original',  latent_z) 
        tf.summary.histogram('z_noise_applied',z_noised) 
        tf.summary.histogram('z_noise',z_noise) 
        merged = tf.summary.merge_all()

        num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
        A_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(A_loss,var_list=var_A, global_step=global_step)      
        D_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=var_D, global_step=global_step)
        G_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)
        H_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(H_loss,var_list=var_H, global_step=global_step)
        clip_epsilon =  e_var.assign(tf.maximum(epsilon_layer,1e-8))
        
        timestamp = str(int(time.time()))        
        if not os.path.exists('results/epsilon_DP/'):
            os.makedirs('results/epsilon_DP/')        
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "results/epsilon_DP/models/{}_".format(dataset) + model_name))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists('results/epsilon_DP/models/'):
            os.makedirs('results/epsilon_DP/models/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists('results/epsilon_DP/dc_out_{}/'.format(dataset)):
            os.makedirs('results/epsilon_DP/dc_out_{}/'.format(dataset))         

        train_writer = tf.summary.FileWriter('results/graphs/epsilon_DP/{}'.format(dataset),sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        if prev_iter != 0:
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))  
        i = prev_iter   
        if prev_iter == 0:
            for idx in range(1):
                if dataset == 'mnist':
                    X_mb, _ = x_train.train.next_batch(mb_size)
                    X_mb = np.reshape(X_mb,[-1,28,28,1])
                elif dataset == 'lsun':
                    X_mb = x_train.next_batch(mb_size)                    
                else:
                    X_mb = next_batch(mb_size, x_train)   
                enc_noise = np.random.laplace(0.0,0.0,[mb_size,z_dim]).astype(np.float32)  
                summary,_,_, A_loss_curr, H_loss_curr = sess.run([merged, A_solver, H_solver, A_loss, H_loss],feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: enc_noise})
                current_step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary,current_step)
                if idx % 100 == 0:
                    print('Iter: {}; A_loss: {:.4}; H_loss: {:.4};'.format(idx,A_loss_curr, H_loss_curr))
                if idx % 1000 == 0: 
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model at {} at step {}'.format(path, current_step))                    
                    
        for idx in range(num_batches_per_epoch):
            if dataset == 'mnist':
                X_mb, _ = x_train.train.next_batch(mb_size)
                X_mb = np.reshape(X_mb,[-1,28,28,1])
            elif dataset == 'lsun':
                X_mb = x_train.next_batch(mb_size)                    
            else:
                X_mb = next_batch(mb_size, x_train) 
            enc_noise = np.random.laplace(0.0,0.0,[mb_size,z_dim]).astype(np.float32)                  
            max_curr, min_curr = sess.run([latent_max,latent_min], feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: enc_noise})
            if idx == 0:
                z_max = max_curr
                z_min = min_curr
            else:
                z_max = np.maximum(z_max,max_curr)
                z_min = np.minimum(z_min,min_curr)
        z_sensitivity = np.abs(np.subtract(z_max,z_min))
        print("Approximated Global Sensitivity:") 
        print(z_sensitivity)        
        z_sensitivity = np.tile(z_sensitivity,(mb_size,1))
        for it in range(num_batches_per_epoch*1000):
            for _ in range(5):
                if dataset == 'mnist':
                    X_mb, _ = x_train.train.next_batch(mb_size)
                    X_mb = np.reshape(X_mb,[-1,28,28,1])
                elif dataset == 'lsun':
                    X_mb = x_train.next_batch(mb_size)
                else:
                    X_mb = next_batch(mb_size, x_train)
                    
                enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: z_sensitivity}) 
            _, H_loss_curr = sess.run([H_solver, H_loss],feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: z_sensitivity})                               
            summary, _, G_loss_curr, dp_epsilon_curr,_ = sess.run([merged, G_solver, G_loss, dp_epsilon,clip_epsilon],feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: z_sensitivity})
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary,current_step)
        
            if it % 100 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; privacy_gain: {:.4}; epsilon: {:.4};'.format(it,D_loss_curr, G_loss_curr,H_loss_curr, dp_epsilon_curr))

            if it % 1000 == 0:    
                Xt_mb = x_test[:mb_size]
                G_sample_curr, re_fake_curr = sess.run([G_sample,G_hacked],feed_dict={X: X_mb, Z_noise: enc_noise, Z_S: z_sensitivity})
                samples_flat = tf.reshape(G_sample_curr,[-1,width,height,channels]).eval()
                img_set = np.append(Xt_mb[:256], samples_flat[:256], axis=0)                  
                samples_flat = tf.reshape(re_fake_curr,[-1,width,height,channels]).eval() 
                img_set = np.append(img_set, samples_flat[:256], axis=0)               

                fig = plot(img_set, width, height, channels)
                plt.savefig('results/epsilon_DP/dc_out_{}/{}.png'.format(dataset,str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model at {} at step {}'.format(path, current_step))