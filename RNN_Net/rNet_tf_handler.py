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
# from net_input_X.aritificial_datasets.bin_seq.bin_seq_1 import gen_bin_seq_epochs as gen_epochs
from net_input_X.aritificial_datasets.bin_seq.bin_seq import gen_bin_seq_epochs as gen_epochs
chars_size=2
# from net_input_X.shakespeare.shakespeare import get_data_info
# gen_epochs, chars_size, idx_to_chars, chars_to_idx = get_data_info() 


# # '''========= ======== ===================='''
# # '''========= net work ===================='''
# # '''========= ======== ===================='''
from rNet_tf import train_network
t = time.time()

from rNet_tf import build_graph
g = build_graph(
	cell_type='GRU',
	state_size=10,
	num_classes=chars_size,
	batch_size=100,
	num_steps=10,
	learning_rate=0.05,
	resetgraph=True)
t = time.time()
training_losses = train_network(g,gen_epochs,num_epochs=1,num_steps=10, batch_size=100, verbose=True, save="net_weights_W/")
print("It took", time.time() - t, "seconds to train for 3 epochs.")
print("The average loss on the final epoch was:", training_losses[-1])

# from rNet_tf import build_multilayer_lstm_graph_with_scan
# g= build_multilayer_lstm_graph_with_scan(
# 	state_size=10,
# 	num_classes=chars_size,
# 	batch_size=100,
# 	num_steps=10,
# 	learning_rate=0.05,
# 	resetgraph=True)
# training_losses = train_network(g,gen_epochs,num_epochs=3,num_steps=10, batch_size=100, verbose=True, save=False)
# print("It took", time.time() - t, "seconds to complete.")


# # # '''========= ======== ===================='''
# # # '''========= testing ===================='''
# # # '''========= ======== ===================='''
# from rNet_tf import generate_characters
# something wrong here
# generate_characters(g, "net_weights_W/", num_chars=750, vocab_to_idx=chars_to_idx)



# # '''========= ======== ===================='''
# # '''========= plotting ===================='''
# # '''========= ======== ===================='''

import matplotlib.pyplot as plt
plt.plot(training_losses)
plt.show()