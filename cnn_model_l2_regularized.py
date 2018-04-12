# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:55:25 2018

@author: Rishi
"""

'''
CNN Model
'''

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
from tensorflow.contrib import learn
from text_cnn_l2_regularized import TextCNN
import data_helpers as dh
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


#================================= Loading data ==============================#

print("Loading data...")

with open('test+train.tsv',encoding='utf8') as tsvfile:
     train_test_data = pd.read_csv(tsvfile, delimiter='\t',header=None)
     tsvfile.close()
     
with open('test.tsv',encoding='utf8') as tsvfile:
    test_data=pd.read_csv(tsvfile, delimiter='\t', header=None)
    tsvfile.close()
    
with open('train.tsv',encoding='utf8') as tsvfile:
    train_data=pd.read_csv(tsvfile, delimiter='\t', header=None)
    tsvfile.close()
    
    
test_text=test_data[2]
Y_test=test_data[1]

train_test_text=train_test_data[2]
Y_train_test=train_test_data[1]
     
train_text=train_data[2]
Y_train=train_data[1]

del test_data, train_test_data, train_data

#============================== Model Parameters =================================#

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",128, " Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")
x_text, y = dh.load_data_and_labels(train_test_text,Y_train_test)
x_1, y_train = dh.load_data_and_labels(train_text,Y_train)
x_2, y_test = dh.load_data_and_labels(test_text,Y_test)


# Build vocabulary
print("building vocab...")
max_document_length =  max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform((x_text))))

x_train = np.array(list(vocab_processor.fit_transform((x_1))))
x_test = np.array(list(vocab_processor.fit_transform((x_2))))


g=tf.Graph()
with g.as_default():
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement= True))
    with sess.as_default():
        
        ''' instantiate the class TextCNN '''
        print("instantiating the class...")
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        
         # Keep track of gradient values and sparsity (optional)
        print("keep track of grad and values....")
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        
        
        
        # Output directory for models and summaries
        print("output directoru for models and summaries...")
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
         
        # Summaries for loss and accuracy
        print("summaries for loss and accuracy...")
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        print("train summaries...")
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Test summaries
        print("test summaries...")
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
         
        
        # Checkpointing
        print("checkpointing...")
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints1"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model1")
        # Tensorflow assumes this directory already exists so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
                
        
        # Write vocabulary
        print("write vocab...")
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        
        print("sess.run...")
        sess.run(tf.initialize_all_variables())
        
        
        print("train_step...")
        def train_step(x_batch, y_batch):
        # a single training step            
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            
            
        print("test step...")   
        def test_step(x_batch, y_batch, writer=None):
            
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
            step, summaries, loss, accuracy = sess.run([global_step, test_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            
        
        print("generate batches...")
        # Generate batches
        batches = dh.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            print("inside loop...")
#            print('batch',batch[1,1])
            x_batch, y_batch = zip(*batch)
#            print('x_batch, y_batch',x_batch, y_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:    
                print("inside if1...")
                print("\nEvaluation:")
                test_step(x_test, y_test, writer=test_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                print("inside if2...")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
