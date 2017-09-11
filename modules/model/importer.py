import ConfigParser as cp

#property of weight
mean = 0
std = 0.3
relu_clip = 20
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 2 * 128
n_hidden_4 = 128
n_hidden_5 = 128
n_hidden_6 = 28

#property of BiRRN LSTM
n_hidden_unit = 1024
forget_bias = 0

#property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
beta1 = 0.9
beta2 = 0.9
epsilon = 1e-6
learning_rate = 0.0001
threshold = 0

def deep_speech(filename):
    config = cp.RawConfigParser()
    config.read(filename)
    name = config.get('init', 'name'),
    layer = config.get('init', 'layer'),
    batch = config.getint('init', 'batch'),
    status = config.get('init', 'status'),
    mean = config.getfloat('init', 'mean'),
    std = config.getfloat('init', 'std')

