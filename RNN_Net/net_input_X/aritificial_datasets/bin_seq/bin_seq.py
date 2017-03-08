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

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_x = np.stack(np.vsplit(raw_x.reshape((-1,num_steps)),batch_size),axis=2)
    data_y = np.stack(np.vsplit(raw_y.reshape((-1,num_steps)),batch_size),axis=2)
    print(data_x.shape)

    for i in range(data_x.shape[0]):
        # print('data_x[i].shape: ',data_x[i].shape)
        yield (data_x[i].T,data_y[i].T)

def gen_bin_seq_epochs(num_epochs, num_steps,batch_size):
    for i in range(num_epochs):
        yield gen_batch(gen_data(), batch_size, num_steps)

# def get_data_info():
#   return gen_bin_seq_epochs, chars_size, idx_to_chars, chars_to_idx
