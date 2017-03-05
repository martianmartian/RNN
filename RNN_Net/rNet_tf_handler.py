#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'

import time
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# ''' ========= special notes =========  '''
# 	contains tf graph for:
# 		basic rnn
# 		multilayer lstm graph with dynamic rnn
# 		multilayer lstm graph with list
# 	contains multiple types of data, hopefully
# ''' ========= end =========  '''


# # '''========= ======== ===================='''
# # '''========= data setup by switching ====='''
# # '''========= ======== ===================='''
from net_input_X.aritificial_datasets.bin_seq.bin_seq import gen_bin_seq_epochs as gen_epochs
chars_size=2
# from net_input_X.shakespeare.shakespeare import get_data_info
# gen_epochs, chars_size, idx_to_chars, chars_to_idx = get_data_info() 


# # '''========= ======== ===================='''
# # '''========= net work ===================='''
# # '''========= ======== ===================='''
from rNet_tf import train_network
t = time.time()
# from rNet_tf import build_multilayer_lstm_graph_with_list
# g = build_multilayer_lstm_graph_with_list(
# from rNet_tf import build_basic_rnn_graph_with_list
# g = build_basic_rnn_graph_with_list(
from rNet_tf import build_multilayer_lstm_graph_with_dynamic_rnn
g = build_multilayer_lstm_graph_with_dynamic_rnn(
	state_size=10,
	num_classes=chars_size,
	batch_size=200,
	num_steps=5,
	learning_rate=0.05,
	resetgraph=True)
print("It took", time.time() - t, "seconds to build the graph.")
training_losses = train_network(g,gen_epochs,num_epochs=3,num_steps=5, batch_size=200, verbose=True, save=False)




# # '''========= ======== ===================='''
# # '''========= plotting ===================='''
# # '''========= ======== ===================='''

import matplotlib.pyplot as plt
plt.plot(training_losses)
plt.show()