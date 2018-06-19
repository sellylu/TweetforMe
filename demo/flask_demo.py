#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)
import numpy as np
from nltk.corpus import wordnet as wn

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/<para>", methods=['GET'])
def index_show(para):
    return redirect(url_for('index'))

@app.route("/write", methods=['POST'])
def write():
    key = request.form['tags']
    if not key:
        key = request.form['recom']
        if not key:
            return redirect(url_for('index'))
    hashtag,_ = GEN.key_similarity(key)
    result = GEN.run(hashtag)
    for k in [key, hashtag]:
        if '#{}'.format(k) not in result: result += ' #{}'.format(k)
        else: result.replace(k, '#{}'.format(k))
    # result = 'Kiss before bed'+' #'+hashtag+b'kissss\xF0\x9F\x98\x8A'.decode()
    return render_template('index.html', result=result, hashtag=key)


class generator():
    def __init__(self):
        self.noise        = np.load('../saved_ncm/prob_ncm.npy')[()]
        self.vocab_to_int = eval(open('../saved_ncm/v2i','r').read())
        self.int_to_vocab = eval(open('../saved_ncm/i2v','r').read())
        self.num_classes  = len(self.vocab_to_int)
        from model import RNNLM
        import tensorflow as tf
        self.model        = RNNLM(self.num_classes)
        saver             = tf.train.Saver()
        self.sess         = tf.Session()
        saver.restore(self.sess, '../saved_model/rnn.ckpt')
        self.PCand        = ['the','i','a','that','how','what','today']
        self.n_net        = {k:wn.synsets(k, pos=wn.NOUN)[:3] for k in self.noise}
    def key_similarity(self, key):
        in_net = wn.synsets(key)[0]
        distance = dict()
        for k,v in self.n_net.items():
            distance[k] = np.mean([in_net.path_similarity(n)/np.sqrt(i+1) for i,n in enumerate(v)])
        return max(distance.items(), key=lambda x: x[1])
    def lm(self, hashtag, PRIME='the'):
        # testtt = dict.fromkeys(list(self.noise)+['original'],'')
        new_state = self.sess.run(self.model.initial_state)
        result = PRIME+' '
        c=self.vocab_to_int[PRIME]
        for _ in range(20):
            feed = {self.model.inputs: np.array([[c]]), self.model.initial_state: new_state}
            prediction, new_state = self.sess.run([self.model.preds, self.model.state], feed_dict=feed)
            # for k in testtt:
            #     if k != 'original':
            #         c = np.argmax(np.log2(np.squeeze(prediction))+self.noise[k])
            #         testtt[k] += self.int_to_vocab[c] + ' '
            #     else:
            #         c = np.argmax(np.log2(np.squeeze(prediction)))
            #         testtt[k] += self.int_to_vocab[c] + ' '
            prob = np.log2(np.squeeze(prediction))+self.noise[hashtag]
            c = np.argmax(prob)
            if (self.int_to_vocab[c] == '<eos>'): break
            result += self.int_to_vocab[c] + ' '
        # from pprint import pprint
        # pprint(testtt)
        return result
    def run(self, key):
        import random
        PRIME = random.choice(self.PCand)
        return self.lm(key,PRIME=PRIME)

if __name__ == "__main__":
    GEN    = generator()
    app.run(debug=True)

