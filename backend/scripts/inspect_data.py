"""
检查processed_data.pkl的数据结构
"""
import pickle
import numpy as np
import os

pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_data.pkl')

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print('Type:', type(data))
if isinstance(data, dict):
    print('Keys:', list(data.keys()))
    for key in data.keys():
        val = data[key]
        print(f'  {key}: type={type(val)}, shape={val.shape if hasattr(val, "shape") else len(val) if hasattr(val, "__len__") else "N/A"}')
elif isinstance(data, (list, tuple, np.ndarray)):
    print('Length:', len(data))
    if len(data) > 0:
        print('First element type:', type(data[0]))
        if hasattr(data[0], 'shape'):
            print('First element shape:', data[0].shape)
        elif isinstance(data[0], dict):
            print('First element keys:', list(data[0].keys()))
