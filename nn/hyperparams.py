# Just a file containing hyperparameter lists.

# Neural network hyper parameters
replacement_types = [replace_mean, replace_zeros]
include_sigma = [True, False]
input_type = ['normal', 'Jy']
input_filters = ['ir', 'uv', 'optical']
output_type = ['median', 'best_fit', 'best_fit_model', 'best_fit_inputs']
output_parameters = range(0, 32)
normalise = ['normalise', 'standardise', 'softmax']

learning_rate = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
decay = [0.00001, 0.0001, 0.001]
loss = ['mae', 'mse', 'rmse', 'msle', 'squared_hinge', 'hinge', 'binary_crossentropy', 'poisson', 'cosine_proximity']

hidden_nodes = [50, 80, 120, 160, 200, 300]
hidden_layers = [1, 2, 3, 4]
batch_size = [100, 250, 500, 1000, 2000, 5000, 10000]
epochs_per_fit = [1, 5, 10, 25, 50, 75, 100]
validation_split = [0.1, 0.2, 0.3]
max_epochs = [100, 200, 500, 750, 1000]
activations = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
use_dropout = [True, False]
dropout_rate = [0.1, 0.25, 0.5, 0.75, 0.9]
initialisation_type = ['uniform', 'lecun_uniform', 'normal', 'identity', 'orthogonal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
