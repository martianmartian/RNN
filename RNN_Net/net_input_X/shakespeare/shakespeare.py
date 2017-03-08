import os
import urllib.request
import numpy as np

"""
Load and process data, utility functions
"""

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = '/Users/martian2049/Desktop/NN:AI/RNN/RNN_Net/net_input_X/shakespeare/tiny-shakespeare.txt'

if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

chars = set(raw_data)
chars_size = len(chars)
idx_to_chars = dict(enumerate(chars))
chars_to_idx = dict(zip(idx_to_chars.values(), idx_to_chars.keys()))

data = [chars_to_idx[c] for c in raw_data]
del raw_data


def ptb_iterator(raw_data, batch_size, num_steps):

  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1] # y is one step forward than x.
    # print('x.shape==>',x.shape)
    # print('x==>',x)
    # print('x_word==>\n',[idx_to_chars[i] for i in x[0]])
    # print('x_word==>\n',[idx_to_chars[i] for i in x[1]])
    # print('x_word==>\n',[idx_to_chars[i] for i in x[2]])
    # print('y==>',y)
    yield (x, y)

def gen_shakespeare_epochs(num_epochs, num_steps, batch_size):
  for i in range(num_epochs):
    yield ptb_iterator(data, batch_size, num_steps)

def get_data_info():
  return gen_shakespeare_epochs, chars_size, idx_to_chars, chars_to_idx
