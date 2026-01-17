"""
检查processed_data.pkl的数据结构
"""
import pickle
import numpy as np
import os

# 更新路径到 data/processed_data.pkl
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(backend_dir)
data_root = os.path.join(project_root, 'data')
pkl_path = os.path.join(data_root, 'processed_data.pkl')

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
