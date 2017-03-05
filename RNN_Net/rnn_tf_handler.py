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


# from net_input_X.aritificial_datasets.bin_seq.bin_seq import gen_bin_seq_epochs




# '''========= shakespeare ========'''
# data =====> 
from net_input_X.shakespeare.shakespeare import get_data_info
gen_shakespeare_epochs, chars_size, idx_to_chars, chars_to_idx = get_data_info() 
# net and graph =====> 
from rnn_tf import train_network
t = time.time()
# from rnn_tf import build_basic_rnn_graph_with_list
# g = build_basic_rnn_graph_with_list(state_size=100,num_classes=chars_size,batch_size=32,num_steps=200,learning_rate=1e-4,resetgraph=True)
# from rnn_tf import build_multilayer_lstm_graph_with_list
# g = build_multilayer_lstm_graph_with_list(state_size=100,num_classes=chars_size,batch_size=32,num_steps=200,learning_rate=1e-4,resetgraph=True)
from rnn_tf import build_multilayer_lstm_graph_with_dynamic_rnn
g = build_multilayer_lstm_graph_with_dynamic_rnn(state_size=100,num_classes=chars_size,batch_size=32,num_steps=200,learning_rate=1e-4,resetgraph=True)
print("It took", time.time() - t, "seconds to build the graph.")
train_network(g,gen_epochs=gen_shakespeare_epochs,num_epochs=1,num_steps=200, batch_size=32, verbose=True, save=False)


