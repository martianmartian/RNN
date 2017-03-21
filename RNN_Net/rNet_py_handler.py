#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'

import time
import numpy as np

# ''' ========= special notes =========  '''
# 	contains tf graph for:
# 		naked rnn with python. 
#     goal is to test basic rnn I/O format...
# ''' ========= end =========  '''


# # '''========= ======== ===================='''
# # '''========= data setup by switching ====='''
# # '''========= ======== ===================='''
from net_input_X.reddit.reddit import get_data_info
X_train,y_train,vocabulary_size,index_to_word,word_to_index = get_data_info()
print(np.array(X_train).shape)
print(np.array(X_train[0]).shape)
print(X_train[0])
# print("Hererere")

# # '''========= ======== ===================='''
# # '''========= net work ===================='''
# # '''========= ======== ===================='''
from rNet_py import RNNNumpy
model = RNNNumpy(
	word_dim=vocabulary_size,
	hidden_dim=10,
	bptt_truncate=4)

# o, s = model.forward_propagation(X_train[2])
losses = model.train_with_sgd(
	X_train[:10],
	y_train[:10],
	learning_rate=0.005, 
	nepoch=3, 
	evaluate_loss_after=1)



# # # '''========= ======== ===================='''
# # # '''========= testing ===================='''
# # # '''========= ======== ===================='''
w_index=[1,2] # at least 2 word
wlist = model.generate_sentence(w_index,word_to_index,index_to_word)
print(wlist)


# # '''========= ======== ===================='''
# # '''========= plotting ===================='''
# # '''========= ======== ===================='''
# import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.show()







