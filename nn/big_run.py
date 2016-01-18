from network_pybrain import run_network
from network import run_network_keras
# Try a really big run.
print 'Starting big run...'
run_network_keras(100, 30, 'mse', None, input_filter_types=None)
print 'Done'
