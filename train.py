#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:36:46 2018

@author: selly
"""
base = '/home/jackchuang/Downloads/nlp/'
data_dir = base + '/emotion-english/data/'
ckpt_dir = base + 'checkpoints_all/'

import re, math
from time import time
from glob import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# read and encode data
words = defaultdict()
train_raw = []
for path in glob(data_dir+'*/*'):
	hashtag = re.findall('\w+',path)[-1]
	raw = re.findall(r'[a-zA-Z<>]+|<eos>', open(path).read().replace('\n',' <eos> '))#[:800000]
	raw.append('<eos>')
	words[hashtag] = Counter(raw)
	train_raw.extend(raw)

vocab = set(train_raw)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
open('v2i','w').write(str(vocab_to_int))
int_to_vocab = dict(enumerate(vocab))
open('i2v','w').write(str(int_to_vocab))
train_encode = np.array([vocab_to_int[c] for c in train_raw], dtype=np.int32)

def noisy(key):
	return np.array([ math.log2(words[key][k]/sum([v[k] for v in words.values()])) if words[key][k]>0 else -np.inf for k in vocab_to_int])


prob_ncm = defaultdict()
for path in glob(data_dir+'*/*'):
	hashtag = re.findall('\w+',path)[-1]
	prob_ncm[hashtag] = noisy(hashtag)
np.save('prob_ncm.npy', prob_ncm)


# Constant
R_CELL      = 'RNN'
batch_size  = 150
time_step   = 20
nodes       = 200
num_layers  = 2
lr          = 0.001
opt         = tf.train.AdamOptimizer #GradientDescentOptimizer(lr)
keep_prob   = 0.5
epochs      = 30
num_classes = len(vocab_to_int)
num_embed   = nodes

def time_reshape(arr, n_seqs, n_steps):
	batch_size = n_seqs*n_steps
	n_batches = len(arr) // batch_size
	arr = arr[:batch_size * n_batches]
	return arr.reshape((n_seqs, -1))
def get_batches(arr, n_seqs, n_steps):
	arr = time_reshape(arr, n_seqs, n_steps)
	for n in range(0, arr.shape[1], n_steps):
		x = arr[:, n:n+n_steps]
		y = np.zeros_like(x)
		y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
		yield x, y
def get_data_array(arr, n_seqs, n_steps):
	x = time_reshape(arr, n_seqs, n_steps).reshape((-1, n_steps))
	y = np.zeros_like(x)
	y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
	return {'x':x,'y':y}
#valid = get_data_array(valid_encode, batch_size, time_step)

#%% structure
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell
def build_inputs(batch_size, time_step, num_classes, num_embed, keep_prob):
    inputs     = tf.placeholder(tf.int32  , shape=(batch_size, time_step), name='input')
#    inputs_hot = tf.one_hot(inputs, num_classes, dtype=tf.int32)
    labels     = tf.placeholder(tf.int32  , shape=(batch_size, time_step), name='labels')
    labels_hot = tf.one_hot(labels, num_classes)
    embedding = tf.get_variable("embedding", [num_classes, num_embed])#, dtype=tf.int32)
    embeded = tf.nn.embedding_lookup(embedding, inputs)
    embeded = tf.nn.dropout(embeded,0.5)
    return inputs, embeded, labels, labels_hot

def build_rnn(in_layer, nodes, num_layers, batch_size, mode='RNN'):
    if mode.upper()=='RNN':
        cell = MultiRNNCell([BasicRNNCell(nodes) for _ in range(num_layers)])
    elif mode.upper()=='LSTM':
        cell = MultiRNNCell([BasicLSTMCell(nodes) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, state = tf.nn.static_rnn(cell, in_layer, initial_state=initial_state)
    return initial_state, outputs, state

def build_output(out_layer, in_size, out_size):
    seq_output = tf.concat(out_layer, 1)
    x = tf.reshape(seq_output, [-1, in_size])
    weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name='dnn/weight')
    bias   = tf.Variable(tf.zeros(out_size), name='dnn/weight')
    logits = tf.matmul(x, weight) + bias
    output = tf.nn.softmax(logits, name='pred')
    return output, logits

def build_loss(logits, labels_hot, num_classes):
    labels = tf.reshape(labels_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss)

def build_accuracy(logits, labels_hot):
    labels = tf.reshape(labels_hot, logits.get_shape())
    correction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correction, tf.float32))

def build_optimizer(loss_op, opt, lr):
    return opt(lr).minimize(loss_op)

#%% bulid graph
tf.reset_default_graph()
inputs, embeded, labels, labels_hot = build_inputs(batch_size, time_step, num_classes, num_embed, keep_prob)
embeded = tf.unstack(embeded, num=time_step, axis=1)
initial_state, outputs, state = build_rnn(embeded, nodes, num_layers, batch_size, R_CELL)

preds, logits = build_output(outputs, nodes, num_classes)
loss_op       = build_loss(logits, labels_hot, num_classes)
acc_op        = build_accuracy(logits, labels_hot)
train_op      = build_optimizer(loss_op, opt, lr)

#%% training
saver = tf.train.Saver(max_to_keep=100)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
iteration = 0
train_loss = list()
tStart = time()
for e in range(epochs):
	new_state = sess.run(initial_state)
	loss, acc = list(), list()
	for x, y in get_batches(train_encode, batch_size, time_step):
		iteration += 1
		feed = {inputs: x, labels: y, initial_state: new_state}
		batch_loss, new_state, _ = sess.run([loss_op, state, train_op], feed_dict=feed)
		loss.append(batch_loss)

		if (iteration%500) == 0:
			print('epochs: {}/{}\titer: {}\tloss: {:.4f}\ttime: {:.1f}(m)'.format(e+1, epochs, iteration, batch_loss, (time()-tStart)/60))

	train_loss.append(np.mean(loss))
	if (e%5)==0:
		saver.save(sess, ckpt_dir+'i{}_l{}.ckpt'.format(iteration, nodes))

saver.save(sess, ckpt_dir+'i{}_l{}.ckpt'.format(iteration, nodes))

sess.close()

#%% Plot figure
#train_i_size = len(train_encode)//(batch_size*time_step)
plt.figure(); plt.plot(train_loss)
plt.xticks(np.arange((epochs)//5+1)*5-1, np.arange((epochs)//5+1)*5);# plt.ylim((0.85,3.1))
plt.title('Learning Curve ({})'.format(R_CELL)); plt.xlabel('epoch'); plt.ylabel('BPC cross-entropy');
#plt.savefig('fig2/loss_{}.jpg'.format(R_CELL))
#print('Training error rate:\t{:.4f}'.format(1-train_acc[-1])

#%% Generation
PRIME='the'

# rebuild model graph
tf.reset_default_graph()
inputs, embeded, labels, labels_hot = build_inputs(1, 1, num_classes, num_embed, 1.0)
embeded = tf.unstack(embeded, num=1, axis=1)
initial_state, outputs, state = build_rnn(embeded, nodes, num_layers, 1, R_CELL)

preds, logits = build_output(outputs, nodes, num_classes)
#
saver = tf.train.Saver()
sess = tf.Session()#config=config)
saver.restore(sess, ckpt_dir+'i{}_l{}.ckpt'.format(iteration, nodes))
#saver.restore(sess, 'model/rnn.ckpt')
new_state = sess.run(initial_state)

result = PRIME+' '
c=vocab_to_int[PRIME]
for n in range(30):
	feed = {inputs: np.array([[c]]), initial_state: new_state}
	prediction, new_state = sess.run([preds, state], feed_dict=feed)
	c = np.argmax(prediction[0])
	if (int_to_vocab[c] == '<eos>'): break
	result += int_to_vocab[c] + ' '

print(result)
#
#sess.close()
