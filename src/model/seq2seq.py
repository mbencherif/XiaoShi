import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model.ops import *
from model.config import Config, Path
from model.features import debed, embed, padding

try:
    df_data = pd.read_pickle(os.path.join(Path.processed_input, "data.pkl"))
except:
    print("data.pkl not found, please run python3 -m model.features first")

x_data = np.array([sample for sample in df_data.trans.values])
y_data = np.array([sample for sample in df_data.origin.values])

print(len(x_data), "poetry data loaded")

class params:
    word_dim = Config.word_dim

    class ec:
        seq_len = Config.trans_seq_len
        h = 100
        layers = 1

    attn_dim = 100

    class dc:
        seq_len = Config.origin_seq_len
        h = 1000
        layers = 1

    batch_size = 50

x = tf.placeholder(tf.float32, [None, params.ec.seq_len, params.word_dim], name="x")

with tf.variable_scope("encoder"):
    ec_outputs, _ = dynamic_rnn("rnn", x, params.ec.h, params.ec.layers)
    ec_outputs_flat = tf.reshape(ec_outputs, [-1, params.ec.seq_len * params.ec.h])
    attn = tf.tanh(linear("attention", ec_outputs_flat, params.attn_dim))

with tf.variable_scope("decoder"):
    dc_outputs, _ = rnn("rnn", attn, params.dc.seq_len, params.dc.h, params.dc.layers)
    dc_outputs_flat = tf.reshape(dc_outputs, [-1, params.dc.h])   
    p_flat = tf.tanh(linear("decoder-result", dc_outputs_flat, params.word_dim))

p = tf.reshape(p_flat, [-1, params.dc.seq_len, params.word_dim], name="p")

y = tf.placeholder(tf.float32, [None, params.dc.seq_len, params.word_dim])

loss = tf.reduce_mean((p - y) ** 2)
tf.summary.scalar('loss', loss)

step = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.InteractiveSession()

writer = tf.summary.FileWriter(os.path.join(Path.logs, "tensorflow"), sess.graph)
merged = tf.summary.merge_all()

tf.global_variables_initializer().run()
saver = tf.train.Saver()

batch_size = params.batch_size

for epoch in range(100000):
    for i in range(len(x_data) // batch_size):
        x_train = x_data[i * batch_size: (i + 1) * batch_size]
        y_train = y_data[i * batch_size: (i + 1) * batch_size]

        _, merged_train, loss_train = sess.run([step, merged, loss], {x: x_train, y: y_train})

    print("epoch", epoch, 'loss', loss_train)

    writer.add_summary(merged_train, epoch)

    if epoch % 100 == 0:

        test_trans = [
            "江南清明时节细雨纷纷飘洒，路上羁旅行人个个落魄断魂。",
            "怎么训练都不收敛，人生已经如此的艰难。",
            "你今天晚上想吃什么？我可以去带你吃。",
            "月亮真好，柳树都开花了。"]

        x_test = np.array([embed(padding(sentence, Config.trans_seq_len)) for sentence in test_trans])
        p_test = p.eval({x: x_test})


        for features in y_train[:2]:
            print(debed(features))

        print()

        for features in p_test:
            print(debed(features))

        saver.save(sess, os.path.join(Path.models,  'model_' + str(epoch) + '.ckpt'))