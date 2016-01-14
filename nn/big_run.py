from network_pybrain import run_network
from network import run_network_keras
# Try a really big run.
print 'Starting big run...'
run_network_keras(500, 20, 'mse', None, input_filter_types=['ir', 'uv', 'optical'])
print 'Done'
