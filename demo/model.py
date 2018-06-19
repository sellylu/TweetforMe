# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:14:50 2018

@author: Selly
"""
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell

class RNNLM():
	def __init__(self, num_classes, num_embed=200):
		tf.reset_default_graph()
		self.inputs, embeded, _, _ = self.build_inputs(1, 1, num_classes, num_embed, 1.0)
		self.initial_state, outputs, self.state = self.build_rnn(embeded, num_embed, 1)
		self.preds, _ = self.build_output(outputs, num_embed, num_classes)

	def build_inputs(self, batch_size, time_step, num_classes, num_embed, keep_prob):
	    inputs     = tf.placeholder(tf.int32  , shape=(batch_size, time_step), name='input')
	    labels     = tf.placeholder(tf.int32  , shape=(batch_size, time_step), name='labels')
	    labels_hot = tf.one_hot(labels, num_classes)

	    embedding = tf.get_variable("embedding", [num_classes, num_embed])#, dtype=tf.int32)
	    embeded = tf.nn.embedding_lookup(embedding, inputs)
	    embeded = tf.nn.dropout(embeded,0.5)
	    return inputs, embeded, labels, labels_hot

	def build_rnn(self, in_layer, nodes, batch_size, num_layers=2, mode='RNN'):
	    if mode.upper()=='RNN':
	        cell = MultiRNNCell([BasicRNNCell(nodes) for _ in range(num_layers)])
	    elif mode.upper()=='LSTM':
	        cell = MultiRNNCell([BasicLSTMCell(nodes) for _ in range(num_layers)])
	    initial_state = cell.zero_state(batch_size, tf.float32)
	    outputs, state = tf.nn.dynamic_rnn(cell, in_layer, initial_state=initial_state)
	    return initial_state, outputs, state

	def build_output(self, out_layer, in_size, out_size):
	    seq_output = tf.concat(out_layer, 1)
	    x = tf.reshape(seq_output, [-1, in_size])

	    weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name='dnn/weight')
	    bias   = tf.Variable(tf.zeros(out_size), name='dnn/weight')

	    logits = tf.matmul(x, weight) + bias
	    output = tf.nn.softmax(logits, name='pred')
	    return output, logits
