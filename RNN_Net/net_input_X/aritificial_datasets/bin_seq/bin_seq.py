import numpy as np

print("\nUsing binary sequence as data !!=01010100110=!!")
print("Expected cross entropy loss if the model:")
print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                      0.375 * np.log(0.375)))
# Learns first dependency only ==> 0.51916669970720941
print("- learns first dependency:  ",
      -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
      -0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches
    # stack them vertically in a data matrix
    n_of_total_batches = data_length // batch_size
    print('n_of_total_batches: ',n_of_total_batches)
    print('batch_size: ',batch_size)
    data_x = np.zeros([batch_size, n_of_total_batches], dtype=np.int32)
    data_y = np.zeros([batch_size, n_of_total_batches], dtype=np.int32)
    print('raw_x[0:15]: ',raw_x[0:15])
    print('raw_x[199:199+15]: ',raw_x[199:199+15])
    for i in range(batch_size):
        print(n_of_total_batches * i,n_of_total_batches * (i + 1))
        data_x[i] = raw_x[n_of_total_batches * i:n_of_total_batches * (i + 1)]
        data_y[i] = raw_y[n_of_total_batches * i:n_of_total_batches * (i + 1)]
    print('data_x[0][0:15]: ',data_x[0][0:15])
    print('data_x[0][199:199+15]: ',data_x[0][199:199+15])
    # for i in range(n_of_total_batches):
    #     data_x[:,i] = raw_x[i*batch_size:(i+1)*batch_size]
    #     data_y[:,i] = raw_y[i*batch_size:(i+1)*batch_size]
    # print('data_x[0][0:15]: ',data_x[0][0:15])
    # print('data_x[0][199:199+15]: ',data_x[0][199:199+15],'\n')
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = n_of_total_batches // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_bin_seq_epochs(num_epochs, num_steps,batch_size):
    for i in range(num_epochs):
        yield gen_batch(gen_data(), batch_size, num_steps)

# def get_data_info():
#   return gen_bin_seq_epochs, chars_size, idx_to_chars, chars_to_idx
