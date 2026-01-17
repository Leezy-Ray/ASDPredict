"""
样本服务
从processed_data.pkl加载样本数据
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple


class SampleService:
    """样本服务类"""
    
    def __init__(self, pkl_path: Optional[str] = None):
        """
        初始化样本服务
        
        Args:
            pkl_path: processed_data.pkl文件路径
        """
        if pkl_path is None:
            # 默认路径：data/processed_data.pkl
            API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            PROJECT_ROOT = os.path.dirname(API_ROOT)
            DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
            pkl_path = os.path.join(DATA_ROOT, 'processed_data.pkl')
        
        self.pkl_path = pkl_path
        self._data = None
        self._samples = None
        
        try:
            self._load_data()
        except Exception as e:
            raise
    
    def _load_data(self):
        """加载pkl数据"""
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Data file not found: {self.pkl_path}")
        
        print(f"Loading data from {self.pkl_path}...")
        with open(self.pkl_path, 'rb') as f:
            self._data = pickle.load(f)
        
        # 解析数据结构
        self._parse_data()
    
    def _parse_data(self):
        """
        解析数据结构
        
        假设processed_data.pkl的结构是：
        - 字典，包含'data'和'labels'键
        - 或列表/数组，每个元素是(sample, label)元组
        - 或其他格式
        
        需要根据实际数据结构调整
        """
        if self._data is None:
            raise ValueError("Data not loaded")
        
        # 尝试多种数据结构
        if isinstance(self._data, dict):
            
            # 处理新格式：{'timeseries': [...], 'labels': [...], 'subject_ids': [...]}
            if 'timeseries' in self._data and 'labels' in self._data:
                timeseries = self._data['timeseries']
                labels = self._data['labels']
                
                # 确保是numpy数组
                if not isinstance(timeseries, np.ndarray):
                    timeseries = np.array(timeseries)
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels)
                
                # timeseries 可能是 (n_subjects, n_windows, window_size, n_rois) 或 (n_subjects, n_timepoints, n_rois)
                # labels 可能是 (n_subjects,) 或 (n_subjects, n_windows)
                # 需要根据 subject_ids 或 labels 的形状来确定
                
                # 如果 labels 是 1D，按 subject 分组
                # 如果 labels 是 2D，需要按 window 处理
                if labels.ndim == 1:
                    # 每个样本一个标签
                    n_subjects = len(labels)
                    self._samples = []
                    for i in range(n_subjects):
                        # 获取第 i 个 subject 的时间序列数据
                        if timeseries.ndim == 3:  # (n_subjects, n_timepoints, n_rois)
                            subject_data = timeseries[i]
                        elif timeseries.ndim == 4:  # (n_subjects, n_windows, window_size, n_rois)
                            # 合并所有窗口
                            windows = timeseries[i]  # (n_windows, window_size, n_rois)
                            subject_data = windows.reshape(-1, windows.shape[-1])  # (n_windows * window_size, n_rois)
                        else:
                            # 尝试按第一维索引
                            subject_data = timeseries[i] if timeseries.ndim >= 2 else timeseries
                        
                        label_val = int(labels[i].item() if hasattr(labels[i], 'item') else labels[i])
                        self._samples.append({
                            'id': i,
                            'data': subject_data,
                            'label': label_val,
                            'type': 'asd' if label_val == 1 else 'control'
                        })
                else:
                    # labels 是 2D 或更高维，需要更复杂的处理
                    raise ValueError(f"Unsupported labels shape: {labels.shape}, expected 1D")
            
            # 字典格式：{'data': [...], 'labels': [...], ...}
            elif 'data' in self._data and 'labels' in self._data:
                data = self._data['data']
                labels = self._data['labels']
                
                # 确保是numpy数组
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels)
                
                # 转换为样本列表
                self._samples = []
                for i in range(len(data)):
                    label_val = labels[i]
                    # 处理 numpy 数组标量
                    if hasattr(label_val, 'item'):
                        label_val = int(label_val.item())
                    else:
                        label_val = int(label_val)
                    
                    self._samples.append({
                        'id': i,
                        'data': data[i],
                        'label': label_val,
                        'type': 'asd' if label_val == 1 else 'control'
                    })
            else:
                # 其他字典格式
                print(f"Warning: Unexpected dict format. Keys: {list(self._data.keys())}")
                # 尝试使用第一个值作为数据
                keys = list(self._data.keys())
                if len(keys) >= 2:
                    data_key = keys[0] if 'data' not in keys else 'data'
                    label_key = keys[1] if 'label' not in keys else 'label'
                    data = self._data[data_key]
                    labels = self._data[label_key]
                    
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    if not isinstance(labels, np.ndarray):
                        labels = np.array(labels)
                    
                    self._samples = []
                    for i in range(len(data)):
                        label_val = labels[i]
                        # 处理 numpy 数组标量
                        if hasattr(label_val, 'item'):
                            label_val = int(label_val.item())
                        else:
                            label_val = int(label_val)
                        
                        self._samples.append({
                            'id': i,
                            'data': data[i],
                            'label': label_val,
                            'type': 'asd' if label_val == 1 else 'control'
                        })
        elif isinstance(self._data, (list, tuple, np.ndarray)):
            # 列表/数组格式：每个元素是(sample, label)或包含标签的样本
            samples_list = list(self._data)
            
            self._samples = []
            for i, item in enumerate(samples_list):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    # (sample, label)格式
                    sample_data = item[0]
                    label = item[1]
                elif isinstance(item, dict):
                    # 字典格式
                    sample_data = item.get('data', item.get('timeseries', item))
                    label = item.get('label', item.get('y', 0))
                else:
                    # 单个样本数据，需要从别处获取标签
                    sample_data = item
                    label = 0  # 默认标签
                
                if not isinstance(sample_data, np.ndarray):
                    sample_data = np.array(sample_data)
                
                self._samples.append({
                    'id': i,
                    'data': sample_data,
                    'label': int(label),
                    'type': 'asd' if int(label) == 1 else 'control'
                })
        else:
            raise ValueError(f"Unsupported data format: {type(self._data)}")
        
        print(f"Loaded {len(self._samples)} samples")
    
    def get_samples(self, n_asd: int = 5, n_control: int = 5) -> List[Dict]:
        """
        获取指定数量的样本（ASD和正常对照组）
        
        Args:
            n_asd: ASD样本数量
            n_control: 正常对照组样本数量
        
        Returns:
            samples: 样本列表，包含original_id用于后续获取数据
        """
        if self._samples is None:
            raise ValueError("Samples not loaded")
        
        # 分离ASD和正常对照组
        asd_samples = [s for s in self._samples if s['type'] == 'asd']
        control_samples = [s for s in self._samples if s['type'] == 'control']
        
        # 随机选择（不固定种子，每次返回不同的随机样本）
        
        if len(asd_samples) < n_asd:
            print(f"Warning: Only {len(asd_samples)} ASD samples available, requested {n_asd}")
            selected_asd = asd_samples
        else:
            indices = np.random.choice(len(asd_samples), n_asd, replace=False)
            selected_asd = [asd_samples[i] for i in indices]
        
        if len(control_samples) < n_control:
            print(f"Warning: Only {len(control_samples)} control samples available, requested {n_control}")
            selected_control = control_samples
        else:
            indices = np.random.choice(len(control_samples), n_control, replace=False)
            selected_control = [control_samples[i] for i in indices]
        
        # 合并并添加metadata
        all_samples = selected_asd + selected_control
        result = []
        
        for idx, sample in enumerate(all_samples):
            # 使用 original_id 作为名称后缀
            original_id = sample['id']
            sample_name = f"{'ASD' if sample['type'] == 'asd' else 'Control'}-{original_id}"
            sample_desc = f"{'自闭症' if sample['type'] == 'asd' else '正常对照组'}样本 #{original_id}"
            
            # #region agent log
            import json
            log_data = {
                "location": "sample_service.py:get_samples",
                "message": "Generating sample metadata",
                "data": {
                    "idx": idx,
                    "original_id": original_id,
                    "type": sample['type'],
                    "generated_name": sample_name,
                    "generated_description": sample_desc
                },
                "timestamp": int(__import__('time').time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H"
            }
            try:
                with open(r'd:\workplace\ASDModelPredict\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            except: pass
            # #endregion
            
            result.append({
                'id': idx,  # 新的ID（用于前端显示和选择）
                'original_id': original_id,  # 原始ID（用于获取数据）
                'type': sample['type'],
                'label': sample['label'],
                'name': sample_name,
                'description': sample_desc
            })
        
        # 保存映射关系，用于后续获取数据
        if not hasattr(self, '_sample_id_map'):
            self._sample_id_map = {}
        for item in result:
            self._sample_id_map[item['id']] = item['original_id']
        
        return result
    
    def get_sample_data(self, sample_id: int) -> Optional[np.ndarray]:
        """
        根据样本ID获取原始数据
        
        Args:
            sample_id: 样本ID（在get_samples返回的列表中）
        
        Returns:
            data: 样本数据数组，shape (n_timepoints, n_rois)
        """
        if self._samples is None:
            raise ValueError("Samples not loaded")
        
        # 如果sample_id在映射中，使用original_id
        if hasattr(self, '_sample_id_map') and sample_id in self._sample_id_map:
            original_id = self._sample_id_map[sample_id]
        else:
            original_id = sample_id
        
        # 找到对应的样本
        for sample in self._samples:
            if sample['id'] == original_id:
                return sample['data']
        
        return None


# 全局服务实例
_sample_service = None


def get_sample_service(pkl_path: Optional[str] = None) -> SampleService:
    """获取样本服务实例（单例）"""
    global _sample_service
    
    if _sample_service is None:
        _sample_service = SampleService(pkl_path)
    return _sample_service


__all__ = ['SampleService', 'get_sample_service']
