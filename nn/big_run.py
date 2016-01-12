from network import run_network_keras
# Try a really big run.

run_network_keras(1000, 50, 'mse', None, True, ['ir', 'uv', 'optical'])
