import os
import numpy as np
import tensorflow as tf

from model.config import Config, Path
from model.features import embed, padding, debed

class Model(object):
    def __init__(self, name):
        try:
            self.sess = tf.Session()
            self.new_saver = tf.train.import_meta_graph(os.path.join(Path.models, name + '.ckpt.meta'))
            # print(self.new_saver.last_checkpoints)
            self.new_saver.restore(self.sess, tf.train.latest_checkpoint(Path.models), )
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name('x:0')
            self.p = graph.get_tensor_by_name('p:0')
            self.disabled = False
        except Exception as e:
            print(e)
            self.disabled = True

    def predict(self, sentence):
        if self.disabled: return "Model not loaded!"
        
        x_test = np.array(embed(padding(sentence, Config.trans_seq_len))).reshape([1, Config.trans_seq_len, Config.word_dim])
        p_test = self.sess.run(self.p, {self.x: x_test})
        return debed(p_test[0])

def main():
    model = Model('model_400')
    print(model.predict("今天，天气真好，柳树都开了花，我非常地思念我的家乡和亲人，但是有点烦"))
    print(model.predict("手拿宝剑，平定万里江山；四海一家，共享道德的涵养"))
    print(model.predict("海角崖山斜成一线，现在也不属于中华之地了"))
    print(model.predict("居庸关上，杜鹃啼鸣，驱马更行，峰回路转，在暮霭四起中，忽遇一带山泉，从峰崖高处曲折来泻，顿令诗人惊喜不已"))
    print(model.predict("现在也不属于中华之地了, 驱马更行，峰回路转，在暮霭四起中"))
    print(model.predict("我在窗前看到一弯明月，想起思念的人的故乡"))
                

if __name__ == '__main__':
    main()