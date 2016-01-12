from network_pybrain import run_network
# Try a really big run.
print 'Starting big run...'
run_network(1000, 50, None, ['ir', 'uv', 'optical'])
print 'Done'
