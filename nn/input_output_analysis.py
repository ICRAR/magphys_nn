import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import numpy as np
import matplotlib.pyplot as plot
from common.database import get_train_test_data, db_init
from normalisation import get_normaliser
import pickle
from binning import *

# Percentile 'cut points'
percentiles_list10 = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
]

percentiles_list100 = range(1, 100)

output_names =[
    'ager',
    'tau_V',
    'agem',
    'tlastb',
    'Mstars',
    'xi_Wtot',
    'sfr29',
    'xi_PAHtot',
    'f_muSFH',
    'fb17',
    'fb16',
    'T_CISM',
    'Ldust',
    'mu_parameter',
    'xi_Ctot',
    'f_muIR',
    'fb18',
    'fb19',
    'T_WBC',
    'SFR_0_1Gyr',
    'fb29',
    'sfr17',
    'sfr16',
    'sfr19',
    'sfr18',
    'tau_VISM',
    'sSFR_0_1Gyr',
    'metalicity_Z_Z0',
    'Mdust',
    'xi_MIRtot',
    'tform',
    'gamma']

input_names =[
    'redshift',
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


def check_temp(filename, config):
    if os.path.exists(filename):
        # read in the header
        with open(filename) as f:
            header = pickle.load(f)

            for key in header:
                if header[key] != config[key]:
                    return False

            return True
    else:
        return False


def load_from_file(filename):
    with open(filename) as f:
        header = pickle.load(f)
        all_in = pickle.load(f)
        all_out = pickle.load(f)
        redshifts = pickle.load(f)
        galaxy_ids = pickle.load(f)

    return all_in, all_out, redshifts, galaxy_ids


def write_file(filename, config, all_in, all_out, redshifts, galaxy_ids):
    with open(filename, 'w') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(redshifts, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(galaxy_ids, f, pickle.HIGHEST_PROTOCOL)


def make_histogram(data, suffix='', name=''):

    plot.clf()

    x = range(0, len(data[0]) - 1)
    i = 1
    for run in data:
        plot.plot(x, run[1:], '-', lw=4)
        print '{0}/{1}'.format(i, len(data))
        i += 1

    plot.xticks(x, input_names[1:], rotation='vertical')

    fig = plot.gcf()
    fig.set_size_inches(10, 10) # 25 25

    plot.savefig(filename='{0}_{1}.png'.format(name, suffix), format='png', dpi=80)


def make_1d_plot(data, drange=(-1, 1), suffix='', name=''):

    percentiles = np.percentile(data, percentiles_list10)
    #percentiles = get_dataspace_cutoffs(data, 0.1)

    plot.clf()
    y = np.zeros_like(data)

    for item in percentiles:
        plot.plot((item, item), (-0.1, 0.1), '-')

    plot.plot(data, y, 'x')
    axes = plot.gca()
    # axes.set_xlim([drange[0],drange[1]])
    axes.set_ylim([-0.1,0.1])

    fig = plot.gcf()
    fig.set_size_inches(25, 6.25)

    plot.savefig(filename='{0}_{1}.png'.format(name, suffix), format='png', dpi=80)


def shuffle_arrays(*args):
    """
    Shuffles all of the given arrays the same way.

    Arrays must be the same length and must be numpy arrays
    """
    if not args:
        # passed nothing
        return

    # Make a random permutation
    perm = np.random.permutation(len(args[0]))

    out_tuple = list()
    for count, item in enumerate(args):
        # Apply permutation to each item, then append them to a list for outputting
        out_tuple.append(item[perm])

    return tuple(out_tuple)


inp = [0,1,2,3,4,5,6,7,8,9,10]
inp2 = [500,700,300,900,400,150,300,700,200,600,800,900,800]

print get_percentile_groups(inp2, 0.1)
print get_dataspace_groups(inp2, 0.1)
exit()

db_init('sqlite:///Database_run06.db')

DatabaseConfig = {'database_connection_string': 'sqlite:///Database_run06.db',
                  'train_data': 200000,
                  'test_data': 1000,
                  'run_id': '06',
                  'output_type': 'median',  # median, best_fit, best_fit_model, best_fit_inputs
                  'input_type': 'normal',  # normal, Jy
                  'include_sigma': False,  # True, False
                  'unknown_input_handler': None,
                  'input_filter_types': None
                  }

if check_temp('nn_last_tmp_input3.tmp', {1:1}):
    all_in, all_out, redshifts, galaxy_ids = load_from_file('nn_last_tmp_input3.tmp')
else:
    all_in, all_out, redshifts, galaxy_ids = get_train_test_data(DatabaseConfig)
    write_file('nn_last_tmp_input3.tmp', {1:1}, all_in, all_out, redshifts, galaxy_ids)


all_in = np.array(all_in)
all_out = np.array(all_out)
redshifts = np.array(redshifts)
galaxy_ids = np.array(galaxy_ids)

all_in, all_out, redshifts, galaxy_ids = shuffle_arrays(all_in, all_out, redshifts, galaxy_ids)

std = get_normaliser('standardise')
standardised = std.normalise(all_in)
norm = get_normaliser('normalise')
normalised = norm.normalise(all_in)
logged = np.log(all_in)
while 'y' != raw_input('Good?'):
    all_in, standardised, normalised, logged = shuffle_arrays(all_in, standardised, normalised, logged)
    make_histogram(all_in[:1])
    make_histogram(standardised[:1], 'std')
    make_histogram(normalised[:1], 'norm')
    make_histogram(logged[:1], 'logged')
exit()



# Shuffle them all in the same way
all_in, all_out, redshifts, galaxy_ids = shuffle_arrays(all_in, all_out, redshifts, galaxy_ids)

print len(np.where(all_in < 0)[0])
print np.shape(all_in)
print np.shape(all_out)
print np.shape(redshifts)
print np.shape(galaxy_ids)

all_in, all_out, redshifts, galaxy_ids = remove_negative_flux(all_in, all_out, redshifts, galaxy_ids)
all_in, all_out, redshifts, galaxy_ids = remove_above_percentile(all_in, all_out, redshifts, galaxy_ids
                                                                 , 99)

print len(np.where(all_in < 0)[0])
print np.shape(all_in)
print np.shape(all_out)
print np.shape(redshifts)
print np.shape(galaxy_ids)

"""
csv_out = []
for i in range(0, len(all_in[0])):

    percentiles = np.percentile(all_in[:,i], percentiles_list100)
    percentiles = np.insert(percentiles, 0, min(all_in[:,i]))
    percentiles = np.append(percentiles, max(all_in[:,i]))

    print input_names[i]
    print percentiles

    row = []
    row.append(input_names[i])
    for item in percentiles:
        row.append(item)

    csv_out.append(row)
    #np.savetxt('{0}.csv'.format(input_names[i]), percentiles, delimiter=',')

for i in range(0, len(all_out[0])):

    percentiles = np.percentile(all_out[:,i], percentiles_list100)
    percentiles = np.insert(percentiles, 0, min(all_out[:,i]))
    percentiles = np.append(percentiles, max(all_out[:,i]))

    print output_names[i]
    print percentiles
    row = []
    row.append(output_names[i])
    for item in percentiles:
        row.append(item)

    csv_out.append(row)
    #np.savetxt('{0}.csv'.format(output_names[i]), percentiles, delimiter=',')

with open('percentiles_all.csv', 'wb') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerows(csv_out)
"""


binned_in = get_dataspace_groups(all_in)
binned_out = get_dataspace_groups(all_out)

for i in range(0, 5):
    print 'Galaxy id: ', galaxy_ids[i]
    print 'Binned in:' ,binned_in[i]
    print 'Binned out:', binned_out[i]
    print
    print all_in[i]
    print all_out[i]
    print
    print

make_histogram(binned_in, name='histogram_binned_10')
exit()

for i in range(0, len(all_in[0])):
    make_1d_plot(all_in[:,i], name=input_names[i])

std = get_normaliser('standardise')
standardised = std.normalise(all_in)
for i in range(0, len(all_in[0])):
    make_1d_plot(standardised[:,i], suffix='std', name=input_names[i])

normaliser = get_normaliser('normalise')
normalised = normaliser.normalise(all_in)
for i in range(0, len(all_in[0])):
    make_1d_plot(normalised[:,i], suffix='norm', name=input_names[i])

normaliser = get_normaliser('softmax')
softmax = normaliser.normalise(all_in)
for i in range(0, len(all_in[0])):
    make_1d_plot(softmax[:,i], suffix='softmax', name=input_names[i])


for i in range(0, len(all_out[0])):
    make_1d_plot(all_out[:,i], name=output_names[i])

std = get_normaliser('standardise')
standardised = std.normalise(all_out)
for i in range(0, len(all_out[0])):
    make_1d_plot(standardised[:,i], suffix='std', name=output_names[i])

normaliser = get_normaliser('normalise')
normalised = normaliser.normalise(all_out)
for i in range(0, len(all_out[0])):
    make_1d_plot(normalised[:,i], suffix='norm', name=output_names[i])

make_histogram(all_in, name='histogram')
make_histogram(standardised, name='histogram_std')
make_histogram(normalised, name='histogram_norm')
make_histogram(softmax, name='histogram_softmax')