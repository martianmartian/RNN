import numpy as np
import tensorflow as tf

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, gen_epochs, num_epochs=1, num_steps=200, batch_size=32, verbose=True, save=False):
    tf.set_random_seed(0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                # print('X.shape==> ',X.shape)
                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

def build_basic_rnn_graph_with_list(
		state_size = 100,
		num_classes = 20,
		batch_size = 32,
		num_steps = 200,
		learning_rate = 1e-4,
		resetgraph=False):

		if resetgraph==True:
				reset_graph()

		x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
		y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

		x_one_hot = tf.one_hot(x, num_classes)
		rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]

		cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
		init_state = cell.zero_state(batch_size, tf.float32)
		# print(init_state)  shape=(32, 100)
		rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

		with tf.variable_scope('softmax'):
				W = tf.get_variable('W', [state_size, num_classes])
				b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
		logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

		y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]
		# print(y_as_list)  shape=(32,)
		loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
		losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
		total_loss = tf.reduce_mean(losses)
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

		return dict(
				x = x,
				y = y,
				init_state = init_state,
				final_state = final_state,
				total_loss = total_loss,
				train_step = train_step
		)

def build_multilayer_lstm_graph_with_list(
		state_size = 100,
		num_classes = 20,
		batch_size = 32,
		num_steps = 200,
		num_layers = 3,
		learning_rate = 1e-4,
		resetgraph=False):

		if resetgraph==True:
				reset_graph()

		x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
		y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

		embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
		rnn_inputs = [tf.squeeze(i) for i in tf.split(1,num_steps, tf.nn.embedding_lookup(embeddings, x))]

		cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
		init_state = cell.zero_state(batch_size, tf.float32)
		rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

		with tf.variable_scope('softmax'):
				W = tf.get_variable('W', [state_size, num_classes])
				b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
		logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

		y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

		loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
		losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
		total_loss = tf.reduce_mean(losses)
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

		return dict(
			x = x,
			y = y,
			init_state = init_state,
			final_state = final_state,
			total_loss = total_loss,
			train_step = train_step
		)

def build_multilayer_lstm_graph_with_dynamic_rnn(
		state_size = 100,
		num_classes = 20,
		batch_size = 32,
		num_steps = 200,
		num_layers = 3,
		learning_rate = 1e-4,
		resetgraph=False):


		if resetgraph==True:
				reset_graph()

		x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
		y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

		embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

		# Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
		rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

		cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
		init_state = cell.zero_state(batch_size, tf.float32)
		rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

		with tf.variable_scope('softmax'):
			W = tf.get_variable('W', [state_size, num_classes])
			b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

		#reshape rnn_outputs and y so we can get the logits in a single matmul
		rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
		y_reshaped = tf.reshape(y, [-1])

		logits = tf.matmul(rnn_outputs, W) + b

		total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

		return dict(
			x = x,
			y = y,
			init_state = init_state,
			final_state = final_state,
			total_loss = total_loss,
			train_step = train_step
		)
