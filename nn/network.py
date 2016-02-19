# Magphys neural network
# Date: 10/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Neural network to perform SED fitting
#
# Input: SED inputs (fuv, nuv, u, g etc.)
# Output: Median fit values
import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from common.database import db_init
from keras.utils.visualize_util import to_graph
from common.logger import config_logger, add_file_handler_to_root
from network_shared import get_training_data, recursive_write_dict, History_Log

from copy import deepcopy

from network_config import read_config, make_default


class FuckingNaN(Exception):
    pass

LOG = config_logger(__name__)
add_file_handler_to_root('nn_run.log')

output_names =[
    'ager',     # 0
    'tau_V',    # 1
    'agem',     # 2
    'tlastb',   # 3
    'Mstars',   # 4
    'xi_Wtot',  # 5
    'sfr29',    # 6
    'xi_PAHtot',# 7
    'f_muSFH',  # 8
    'fb17',     # 9
    'fb16',     # 10
    'T_CISM',   # 11
    'Ldust',    # 12
    'mu_parameter', # 13
    'xi_Ctot',  # 14
    'f_muIR',   # 15
    'fb18',     # 16
    'fb19',     # 17
    'T_WBC',    # 18
    'SFR_0_1Gyr',# 19
    'fb29',     # 20
    'sfr17',    # 21
    'sfr16',    # 22
    'sfr19',    # 23
    'sfr18',    # 24
    'tau_VISM', # 25
    'sSFR_0_1Gyr', # 26
    'metalicity_Z_Z0', # 27
    'Mdust',    # 28
    'xi_MIRtot',# 29
    'tform',    # 30
    'gamma'     # 31
]

input_names = [
    'fuv',
    'nuv',
    'u',
    'g',
    'r',
    'i',
    'z',
    'Z',
    'Y',
    'J',
    'H',
    'K',
    'WISEW1',
    'WISEW2',
    'WISEW3',
    'WISEW4',
    'PACS100',
    'PACS160',
    'SPIRE250',
    'SPIRE350',
    'SPIRE500'
]

optimiser_names = [
    'sgd',
    'adagrad',
    'adadelta',
    'adam',
    'rmsprop'
]


def autoencoder_test(train_in, test_in):

    model = Sequential()

    model.add(Dense(output_dim=15, input_dim=42))
    model.add(PReLU())
    model.add(Dense(output_dim=22, input_dim=15))

    model.compile(loss='mse', optimizer=RMSprop(lr=0.001), class_mode='binary')

    model.fit(train_in, train_in, 5000, 10000, validation_split=0.3, verbose=True, show_accuracy=True)

    for i in range(0, 10):
        ans = model.predict(np.array([test_in[i]]))

        print 'Test', test_in[i]
        print 'Ans', ans[0]
        print
        print

    exit()


def save_differences_graph(differences, filename, epochs_per_entry):
    plt.clf()
    x = range(0, len(differences) * epochs_per_entry, epochs_per_entry)
    y = differences

    print x
    print y

    print np.shape(x)
    print np.shape(y)
    plt.plot(x, y, label='differences', linewidth=2)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('differences')
    plt.xlabel('epochs')

    plt.savefig(filename='{0}.png'.format(filename), format='png')


def save_graph(history, filename, epochs_per_entry):
    """
    Saves a neural network loss graph. With loss and validation loss over epoch
    :param history:
    :param filename:
    :param epochs_per_entry:
    :return:
    """
    plt.clf()
    x = range(0, len(history) * epochs_per_entry, epochs_per_entry)
    y1 = []
    y2 = []

    for item in history:
        y1.append(item['loss'])
        y2.append(item['val_loss'])

    plt.plot(x, y1, label='loss', linewidth=2)
    plt.plot(x, y2, label='val_loss', linewidth=2)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('error rate')
    plt.xlabel('epochs')

    plt.savefig(filename='{0}.png'.format(filename), format='png')


def save_mean_convergence_graph(filename, values, means, mins, maxs, epochs_per_entry):
    # Values (nb_epochs, 32)
    # Means (32,)

    # Make a graph showing differences from mean over time

    x = range(0, len(values) * epochs_per_entry, epochs_per_entry)

    for a in range(0, len(values[0])):
        for i in range(0, len(values[0][0])):
            plt.clf()
            plt.plot(x, values[:,a,i])

            try:
                plt.plot((0, len(values) * epochs_per_entry), (means[i], means[i]), 'k-', color='red', lw=1)
                plt.plot((0, len(values) * epochs_per_entry), (mins[i], mins[i]), 'k-', color='blue', lw=4)
                plt.plot((0, len(values) * epochs_per_entry), (maxs[i], maxs[i]), 'k-', color='blue', lw=4)
                plt.savefig(filename='{0}_{1}_test-{2}.png'.format(filename, output_names[i], a), format='png')
            except IndexError:
                plt.plot((0, len(values) * epochs_per_entry), (means, means), 'k-', color='red', lw=1)
                plt.plot((0, len(values) * epochs_per_entry), (mins, mins), 'k-', color='blue', lw=4)
                plt.plot((0, len(values) * epochs_per_entry), (maxs, maxs), 'k-', color='blue', lw=4)
                plt.savefig(filename='{0}_test-{2}.png'.format(filename, output_names[i], a), format='png')


def do_tests(model, test_in, num_tests):

    # Randomly generate 30 tests to do
    tests_to_use = np.random.randint(0, len(test_in) - 1, size=num_tests)

    aves = []
    tests = []
    for test in tests_to_use:  # Do 30 tests.
        this_test = {}

        ans = model.predict(np.array([test_in[test]]))

        aves.append(ans[0])

        this_test['test_id'] = test
        this_test['output'] = ans

        tests.append(this_test)

    means = np.mean(aves, axis=0)
    std = np.std(aves, axis=0)

    return tests, means, std


def write_test_data(f,
                    test_results,
                    test_in,
                    test_out,
                    test_means,
                    test_std,
                    galaxy_ids_test,
                    out_normaliser,
                    single_output):

    # Write the test data out.
    for item in test_results:
        test_id = item['test_id']
        ans = item['output']

        f.write('\nGalaxy number = {0}\n'.format(galaxy_ids_test[test_id]))

        if config['PreprocessingConfig']['normalise_input'] is not None:
            f.write('Normalised inputs: \n')
        else:
            f.write('Inputs: \n')

        for i, inp in enumerate(test_in[test_id]):

            f.write('{0}: \t{1}\n'.format(i, inp))

        f.write('\nName\tOutput\t\tCorrect\n')

        # If we're doing single values, we'll get a scalar here for test_out so indexing it
        # causes a crash. Need this code to differentiate
        try:
            test_vals = [test_out[test_id]]
        except TypeError:
            test_vals = test_out

        for i, (predicted, correct) in enumerate(zip(ans[0], test_vals)):
            if out_normaliser is not None:
                predicted = out_normaliser.denormalise_value(predicted, i)
                correct = out_normaliser.denormalise_value(correct, i)

            if single_output is not None:
                i = single_output

            f.write('{0}: \t{1}\t\t{2}\n'.format(output_names[i], predicted, correct))

    f.write('Mean, std for each network output over these tests:\n')
    for i, (m, s) in enumerate(zip(test_means, test_std)):
        if single_output is not None:
            i = single_output
        f.write('{0}: {1}\t{2}'.format(output_names[i], m, s))


def write_summary_data(f,
                       config,
                       evaluation,
                       epoch_data,
                       mean_in,
                       stddev_in,
                       mean_out,
                       stddev_out,
                       val_loss
                       ):

    # Write overall score
    f.write('OVERALL SCORE ON TEST SETS\n')
    recursive_write_dict(f, evaluation)

    f.write('\nLowest validation loss on validation set {0} at epoch {1}\n'.format(val_loss[1], val_loss[0]))

    # Write network configuration
    f.write('\n\nConfiguration\n')
    recursive_write_dict(f, config)

    # Write loss, val loss and accurary each epoch
    f.write('\n\nEpoch data\n')
    for i, item in enumerate(epoch_data):
        f.write('{0}:   {1}\n'.format(i, item))

    # Write means and stddev for the input.
    f.write('\n\nMean\tstddev in\n')
    try:  # Try to treat as iterable. If it's not, then treat as scalar
        for i, (m, s) in enumerate(zip(mean_in, stddev_in)):
            f.write('Input {0}:\t{1}\t{2}\n'.format(i, m, s))
    except TypeError:
        f.write('Input {0}:\t{1}\t{2}\n'.format(0, mean_in, stddev_in))

    # Write means and stddev for the output.
    f.write('\n\nMean\tstddev out\n')
    try:
        for i, (m, s) in enumerate(zip(mean_out, stddev_out)):
            f.write('Output {0}:\t{1}\t{2}\n'.format(i, m, s))
    except TypeError:
        f.write('Output {0}:\t{1}\t{2}\n'.format(0, mean_out, stddev_out))


def build_network(net_structure, input_dim, output_dim, optimiser):
    model = Sequential()
    model.add(Dense(output_dim=net_structure['hidden_connections'], input_dim=input_dim, init=net_structure['initialisation_type'], activation=net_structure['input_activation']))

    if net_structure['use_dropout']:
        model.add(Dropout(net_structure['dropout_rate']))
        LOG.info('Using Dropout')

    if net_structure['use_batch_norm']:
        model.add(BatchNormalization())
        LOG.info('Using batch normalisation')

    for i in range(0, net_structure['hidden_layers']):
        model.add(Dense(output_dim=net_structure['hidden_connections'], input_dim=net_structure['hidden_connections'], init=net_structure['initialisation_type'], activation=net_structure['hidden_activation']))

        if net_structure['use_dropout']:
            model.add(Dropout(net_structure['dropout_rate']))

        if net_structure['use_batch_norm']:
            model.add(BatchNormalization())

    model.add(Dense(output_dim=output_dim, input_dim=net_structure['hidden_connections'], init=net_structure['initialisation_type'], activation=net_structure['output_activation']))
    model.compile(loss=net_structure['loss'], optimizer=optimiser)

    return model


def weights_to_list(model):
    return [l.get_weights() for l in model.layers]


def weight_from_list(model, weights):
    if len(model.layers) != len(weights):
        raise Exception('Model has different length to weights list: Model {0} vs List {1}'
                        .format(len(model.layers), len(weights)))

    for l, w in zip(model.layers, weights):
        l.set_weights(w)

    return model

db_init('sqlite:///Database_run06.db')


def run_network_keras(config, output_file_name=None):

    for line in config:
        print line, config[line]
        print

    # Unpack configs required
    learning_params = config['LearningParameters']
    net_structure = config['NetworkStructure']
    fitting_params = config['TrainingParameters']
    file_config = config['FileConfig']

    out_directory = file_config['output_directory']

    if output_file_name is None:  # Use the config files name only if we're not provided one here.
        output_file_name = file_config['output_file_name']

    train_data = get_training_data(config['DatabaseConfig'], config['PreprocessingConfig'], config['FileConfig'])

    # Training data comes packed in to dictionaries to avoid needing to return a whole pile of values.
    train_in = train_data['train_in']
    train_out = train_data['train_out']
    test_in = train_data['test_in']
    test_out = train_data['test_out']
    galaxy_ids_test = train_data['galaxy_ids_test']
    out_normaliser = train_data['out_normaliser']

    # Some general statistics of the dataset
    mean_in = train_data['mean_in']
    mean_out = train_data['mean_out']
    stddev_in = train_data['stddev_in']
    stddev_out = train_data['stddev_out']
    min_out = train_data['min_out']
    max_out = train_data['max_out']

    LOG.info('Dataset shape train')
    LOG.info('{0}'.format(np.shape(train_in)))
    LOG.info('{0}'.format(np.shape(train_out)))
    LOG.info('Dataset shape test')
    LOG.info('{0}'.format(np.shape(test_in)))
    LOG.info('{0}'.format(np.shape(test_out)))

    LOG.info('Compiling neural network model')

    if net_structure['optimiser'] == 'sgd':
        optimiser = SGD(lr=learning_params['learning_rate'], momentum=learning_params['momentum'], decay=learning_params['decay'], nesterov=True)
    else:
        optimiser = net_structure['optimiser']

    print optimiser

    input_dim = len(train_in[0])
    try:
        output_dim = len(train_out[0])
    except:
        output_dim = 1

    model = build_network(net_structure, input_dim, output_dim, optimiser)

    LOG.info("Compiled.")

    # Train the model each generation and show predictions against the validation dataset
    history_log = History_Log()
    trained = False
    total_epoch = 0

    epoch_history = []
    differences = []

    lowest_val_loss = 99999
    lowest_val_loss_weights = None  # Best weight configuration with lowest validation loss
    lowest_val_loss_epoch = 0

    while not trained:

        LOG.info('epoch {0} / {1}'.format(total_epoch, fitting_params['max_epochs']))

        model.fit(train_in, train_out, batch_size=fitting_params['batch_size'],
                  nb_epoch=fitting_params['epochs_per_fit'],
                  validation_split=fitting_params['validation_split'],
                  show_accuracy=True, verbose=False, callbacks=[history_log])

        LOG.info('{0}'.format(history_log.epoch_data))
        total_epoch += fitting_params['epochs_per_fit']

        if np.isnan(history_log.epoch_data['val_loss']):
            raise FuckingNaN("Nan'd")

        # If the val loss is lower, save the weights
        if history_log.epoch_data['val_loss'] < lowest_val_loss:
            # We have something with lower validation loss.
            lowest_val_loss = history_log.epoch_data['val_loss']
            lowest_val_loss_weights = weights_to_list(model)
            lowest_val_loss_epoch = total_epoch

        # Predict a test sample (change 1 to any other value to test more than 1)
        # and use it to track how the network's output for this/these test(s) changes over time.
        prediction = model.predict(np.array(test_in[:3]))
        differences.append(prediction)

        epoch_history.append(history_log.epoch_data)

        if history_log.epoch_data['val_loss'] < 0.001 or total_epoch >= fitting_params['max_epochs']:
            trained = True

    differences = np.array(differences)  # Need special np indexing on this later

    if not out_directory:
        out_directory = os.getcwd()

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    if not os.path.exists('{0}/graph'.format(out_directory)):
        os.mkdir('{0}/graph'.format(out_directory))

    print '{0}/graph'.format(out_directory)

    graph_out = '{0}/graph/{1}'.format(out_directory, output_file_name)

    # Save network weights, graph of network and loss over epoch graph.
    to_graph(model).write_svg("{0}_Graph.svg".format(graph_out))
    model.save_weights('{0}_weights.h5'.format(graph_out), overwrite=True)
    save_mean_convergence_graph('{0}_convergence'.format(graph_out),differences, mean_out, min_out, max_out, fitting_params['epochs_per_fit'])
    save_graph(epoch_history, '{0}_loss'.format(graph_out), fitting_params['epochs_per_fit'])

    # do 30 tests on the network's final weights.
    test_results, test_means, test_std = do_tests(model, test_in, fitting_params['num_tests'])
    # evaluate the network on its final weights
    evaluation = {'At end of training:': model.evaluate(test_in, test_out, 1000, True, True)}

    # do 30 tests on the lowest validation loss weights.
    model = weight_from_list(model, lowest_val_loss_weights)
    model.save_weights('{0}_best_weights.h5'.format(graph_out), overwrite=True)
    val_test_results, val_test_means, val_test_std = do_tests(model, test_in, fitting_params['num_tests'])
    # evaluate the network on the lowest validation loss weights
    evaluation['Best validation loss:'] = model.evaluate(test_in, test_out, 1000, True, True)

    with open('{0}/{1}.txt'.format(out_directory, output_file_name), 'w') as f:
        write_summary_data(f,
                           config,
                           evaluation,
                           epoch_history,
                           mean_in,
                           stddev_in,
                           mean_out,
                           stddev_out,
                           (lowest_val_loss_epoch, lowest_val_loss)
                           )

        f.write('\n\n\n\n----------TEST DATA FOR FINAL MODEL----------\n\n\n\n')
        write_test_data(f,
                        test_results,
                        test_in,
                        test_out,
                        test_means,
                        test_std,
                        galaxy_ids_test,
                        out_normaliser,
                        config['PreprocessingConfig']['single_output']
                        )
        f.write('\n\n\n\n----------TEST DATA FOR BEST VALIDATION LOSS MODEL----------\n\n\n\n')
        write_test_data(f,
                        val_test_results,
                        test_in,
                        test_out,
                        val_test_means,
                        val_test_std,
                        galaxy_ids_test,
                        out_normaliser,
                        config['PreprocessingConfig']['single_output'])

if __name__ == '__main__':

    config = None
    if len(sys.argv) < 2:
        if os.path.isfile('{0}/{1}'.format(os.getcwd(), 'default.cfg')):
            config = read_config('default.cfg')
        else:
            make_default('default.cfg')
            LOG.info('No config file specified. Making a default in {0}'.format(os.getcwd()))
    else:
        if not os.path.isfile(sys.argv[1]):
            LOG.info('Config file with the name {0} does not exist'.format(sys.argv[1]))
        else:
            config = read_config(sys.argv[1])

    if not config:
        exit()

    default_config = deepcopy(config)


    # 3 runs, one normal, one with replace mean, one with replace zeros
    """
    input_handlers = ['replace_mean']
    for item in input_handlers:
        config['DatabaseConfig']['unknown_input_handler'] = item
        config['FileConfig']['output_directory'] = 'invalid_input_handlers'
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(item)
        run_network_keras(config)
    config = deepcopy(default_config)

    # 3 runs, one with no normalisation, one with minmax, one with standardise
    normalisations = ['normalise', 'standardise', None]
    for item in normalisations:
        config['PreprocessingConfig']['normalise_input'] = item
        config['FileConfig']['output_directory'] = 'normalisaions'
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(item)
        run_network_keras(config)
    config = deepcopy(default_config)

    # 4 * 4runs. With 4 variations of learning parameters
    lr = [0.01, 0.005, 0.02]
    momentum = [0.0, 0.25, 0.5, 0.75]
    for item in lr:
        for item2 in momentum:
            config['LearningParameters']['learning_rate'] = item
            config['LearningParameters']['momentum'] = item2
            config['FileConfig']['output_directory'] = 'learning_params'
            config['FileConfig']['output_file_name'] = 'output_l{0}_m{1}'.format(item, item2)
            run_network_keras(config)
    config = deepcopy(default_config)

    # 2 runs, one with sigmas and one without sigmas
    for item in [True, False]:
        config['DatabaseConfig']['include_sigma'] = item
        config['FileConfig']['output_directory'] = 'include_sigma'
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(item)
        run_network_keras(config)
    config = deepcopy(default_config)

    # 2 runs, one using Jy input and one using normal input
    for item in ['Jy', 'normal']:
        config['DatabaseConfig']['input_type'] = item
        config['FileConfig']['output_directory'] = 'input_type'
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(item)
        run_network_keras(config)
    config = deepcopy(default_config)

    # 10 runs, each using a different network initialisation type
    inits = ['orthogonal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    for item in inits:
        config['NetworkStructure']['initialisation_type'] = item
        config['FileConfig']['output_directory'] = 'initialisations'
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(item)
        run_network_keras(config)
    config = deepcopy(default_config)

    # 3 * 3 runs. Test each binning type against each other binning type, along with no binning.
    b_types = ['percentile', 'absolute']
    b_precision = [0.1, 0.01]
    for item in b_types:
        for item2 in b_precision:
            config['PreprocessingConfig']['binning_type'] = item
            config['PreprocessingConfig']['binning_precision'] = item2
            config['FileConfig']['output_directory'] = 'binning'
            config['FileConfig']['output_file_name'] = 'output_{0}_{1}'.format(item, item2)
            run_network_keras(config)
    config = deepcopy(default_config)
    """


    #losses = ['mae']
    #activations = ['relu', 'tanh']
    #config['NetworkStructure']['loss'] = 'rmse'
    # including sigmas
    """
    for item in single_outputs:
        config['DatabaseConfig']['input_filter_types'] = ['ir']
        config['PreprocessingConfig']['single_output'] = item
        config['DatabaseConfig']['include_sigma'] = True
        config['FileConfig']['output_directory'] = 'successful_{0}'.format('sigma')
        config['FileConfig']['output_file_name'] = 'output_{0}'.format(output_names[item])
        run_network_keras(config)
    config = deepcopy(default_config)
    exit()


    # Removing percentiles
    percentiles = [90, 99]
    for item in single_outputs:
        for item2 in percentiles:
            config['DatabaseConfig']['input_filter_types'] = ['ir']
            config['PreprocessingConfig']['single_output'] = item
            config['PreprocessingConfig']['erase_above'] = item2
            config['FileConfig']['output_directory'] = 'successful/remove_percentile/{0}'.format(item2)
            config['FileConfig']['output_file_name'] = 'output_{0}'.format(output_names[item])
            run_network_keras(config)
    config = deepcopy(default_config)

    # single_outputs = [4, 12, 28]

    single_outputs = [12, 28]
    # Different optimisers
    for item in single_outputs:
        for optimiser in optimiser_names:
            config['DatabaseConfig']['input_filter_types'] = ['ir']
            config['PreprocessingConfig']['single_output'] = item
            config['NetworkStructure']['optimiser'] = optimiser
            config['FileConfig']['output_directory'] = 'successful/optimisers/{0}'.format(optimiser)
            config['FileConfig']['output_file_name'] = 'output_{0}_{1}'.format(optimiser, output_names[item])
            run_network_keras(config)
    config = deepcopy(default_config)
    """

    single_outputs = [4, 12, 28]
    # replace_mean and replace_zeros
    """
    replacements = ['replace_mean','replace_zeros']
    for item2 in replacements:
        for item in single_outputs:
            config['DatabaseConfig']['input_filter_types'] = ['ir']
            config['PreprocessingConfig']['single_output'] = item
            config['DatabaseConfig']['unknown_input_handler'] = item2
            config['FileConfig']['output_directory'] = 'successful/replace/{0}'.format(item2)
            config['FileConfig']['output_file_name'] = 'output_{0}'.format(output_names[item])
            run_network_keras(config)
    config = deepcopy(default_config)


    with_sigma = [False,True]
    for item1 in with_sigma:
        for item2 in single_outputs:
            config['PreprocessingConfig']['single_output'] = item2
            config['DatabaseConfig']['include_sigma'] = item2
            config['DatabaseConfig']['input_filter_types'] = ['ir']
            config['FileConfig']['output_directory'] = 'successful/sigmas/{0}'.format(item1)
            config['FileConfig']['output_file_name'] = 'output_{0}'.format(output_names[item2])
            run_network_keras(config)
    config = deepcopy(default_config)

    """
    """


    single_outputs = [4, 12, 28]
    # Different optimisers
    for item in single_outputs:
        for optimiser in optimiser_names:
            config['DatabaseConfig']['input_filter_types'] = ['ir']
            config['PreprocessingConfig']['single_output'] = item
            config['NetworkStructure']['optimiser'] = optimiser
            config['FileConfig']['output_directory'] = 'successful/optimisers/{0}'.format(optimiser)
            config['FileConfig']['output_file_name'] = 'output_{0}_{1}'.format(optimiser, output_names[item])
            run_network_keras(config)
    config = deepcopy(default_config)
    """
    single_outputs = [4, 12, 28]
    # 3 * 3 runs. Test each binning type against each other binning type, along with no binning.
    b_types = ['percentile']
    b_precision = [0.01]
    for item1 in single_outputs:
        for item in b_types:
            for item2 in b_precision:
                config['PreprocessingConfig']['binning_type'] = item
                config['PreprocessingConfig']['binning_precision'] = item2
                config['PreprocessingConfig']['single_output'] = item1
                config['FileConfig']['output_directory'] = 'binning'
                config['DatabaseConfig']['input_filter_types'] = ['ir']
                config['FileConfig']['output_directory'] = 'successful/bin/{0}'.format(item2)
                config['FileConfig']['output_file_name'] = 'output_{0}'.format(output_names[item1])
            run_network_keras(config)
    config = deepcopy(default_config)

    exit()

    input_types = [['uv', 'optical'], ['ir', 'optical'], None]
    single_outputs = range(17, 32)
    for item in input_types:
        if item == ['uv', 'optical']:
            single_outputs = range(17, 32)
        else:
            single_outputs = range(0, 32)
        for item2 in single_outputs:
            config['DatabaseConfig']['input_filter_types'] = item
            config['PreprocessingConfig']['single_output'] = item2
            config['FileConfig']['output_directory'] = 'filter_types/{0}'.format(item)
            config['FileConfig']['output_file_name'] = 'output_{0}_{1}'.format(item, output_names[item2])

            success = False
            NaNcount = 0

            while not success:
                try:
                    run_network_keras(config)
                    success = True
                except FuckingNaN:
                    if NaNcount >= 5:
                        print 'Forget it.'
                        continue
                    print 'One of those fucking NaNs occurred. Do this shit again.'
                    NaNcount += 1

    LOG.info("Done")

    """
    # Config for the current run
    fitting_params['max_epochs'] = 100000
    fitting_params['epochs_per_fit'] = 100
    fitting_params['batch_size'] = 1000
    net_structure['input_activation'] = 'relu'
    net_structure['hidden_activation'] = 'relu'
    net_structure['output_activation'] = 'linear'
    net_structure['use_dropout'] = True

    tmp_file = 'nn_last_tmp_input1.tmp'

    output_dir = 'long/'
    os.mkdir('long')
    """