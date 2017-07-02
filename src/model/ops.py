import tensorflow as tf

def linear(scope_name, x, output_dim, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        w = tf.get_variable('w', [x.shape[-1], output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_dim], tf.float32)
        tf.summary.histogram('w', w)
        tf.summary.histogram('b', b)
        y = tf.matmul(x, w) + b
        return y

def dynamic_rnn(scope_name, x, hidden_dim, n_layers=1, initial_state=None, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        cells = [tf.contrib.rnn.BasicLSTMCell(hidden_dim) for i in range(n_layers)]
        stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(stacked_cells, x, dtype=tf.float32, scope=scope, initial_state=initial_state)
        return outputs, state

def rnn(scope_name, z, seq_len, hidden_dim, n_layers=1):
    with tf.variable_scope(scope_name) as scope:
        cells = [tf.contrib.rnn.GRUCell(hidden_dim) for i in range(n_layers)]
        stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
        state = stacked_cells.zero_state(tf.shape(z)[0], tf.float32)
        x = tf.zeros([tf.shape(z)[0], hidden_dim])
        outputs = []
        for i in range(seq_len):
            z_x = tf.concat([z, x], axis=1)
            x, state = stacked_cells(z_x, state)
            scope.reuse_variables()
            outputs.append([x])
        return tf.transpose(tf.concat(outputs, axis=0), [1, 0, 2]), state
