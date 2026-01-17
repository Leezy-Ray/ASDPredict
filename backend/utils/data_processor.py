"""
数据处理器
处理用户输入的fMRI ROI数据，进行窗口切分和PCC计算
"""

import os
import sys
import numpy as np
# 延迟导入 pandas，避免与 numpy 2.x 的兼容性问题
# import pandas as pd
import json
from typing import Dict, List, Tuple, Union

# API根目录（用于相对导入）
API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_ROOT)


def load_fmri_data_from_json(data: Union[str, dict]) -> np.ndarray:
    """
    从JSON加载fMRI数据
    
    Args:
        data: JSON字符串或字典，格式：
            - {"timeseries": [[...], [...]]}  # (n_timepoints, n_rois)
            - {"data": [[...], [...]]}  # 同上
            - [[...], [...]]  # 直接是列表
    
    Returns:
        timeseries: ndarray, shape (n_timepoints, n_rois) 或 (n_rois, n_timepoints)
    """
    if isinstance(data, str):
        data = json.loads(data)
    
    # 尝试不同的键名
    if isinstance(data, dict):
        if 'timeseries' in data:
            timeseries = np.array(data['timeseries'], dtype=np.float32)
        elif 'data' in data:
            timeseries = np.array(data['data'], dtype=np.float32)
        elif 'fmri' in data:
            timeseries = np.array(data['fmri'], dtype=np.float32)
        else:
            raise ValueError("JSON中找不到timeseries/data/fmri键")
    elif isinstance(data, list):
        timeseries = np.array(data, dtype=np.float32)
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    # 检查并转置：确保是(n_timepoints, n_rois)
    if timeseries.shape[0] < timeseries.shape[1]:
        # 如果时间点少于ROI数，可能是(n_rois, n_timepoints)，需要转置
        timeseries = timeseries.T
    
    return timeseries


def load_fmri_data_from_csv(csv_path: str) -> np.ndarray:
    """
    从CSV加载fMRI数据
    
    Args:
        csv_path: CSV文件路径
            格式1: 每行是一个时间点，每列是一个ROI
            格式2: 每行是一个ROI，每列是一个时间点
    
    Returns:
        timeseries: ndarray, shape (n_timepoints, n_rois)
    """
    # 延迟导入 pandas
    import pandas as pd
    df = pd.read_csv(csv_path, header=0)
    
    # 移除可能的索引列
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    timeseries = df.values.astype(np.float32)
    
    # 判断是否需要转置：通常CSV中时间点少（100），ROI多（200）
    if timeseries.shape[0] < timeseries.shape[1]:
        timeseries = timeseries.T
    
    return timeseries


def align_to_cc200(timeseries: np.ndarray, input_rois: int = None) -> np.ndarray:
    """
    将输入的ROI数据对齐到CC200（200个ROI）
    
    Args:
        timeseries: ndarray, shape (n_timepoints, n_rois)
        input_rois: 输入的ROI数量（如果为None则自动检测）
    
    Returns:
        aligned_timeseries: ndarray, shape (n_timepoints, 200)
    """
    n_timepoints, n_rois = timeseries.shape
    
    if input_rois is None:
        input_rois = n_rois
    
    target_rois = 200  # CC200标准
    
    if n_rois == target_rois:
        # 已经是200个ROI，直接返回
        return timeseries
    elif n_rois < target_rois:
        # ROI数少于200，需要填充
        # 使用零填充或重复最后一个ROI
        padding = np.zeros((n_timepoints, target_rois - n_rois), dtype=np.float32)
        aligned = np.concatenate([timeseries, padding], axis=1)
        return aligned
    else:
        # ROI数多于200，需要选择或降维
        # 方案1: 选择前200个ROI
        # 方案2: 选择均匀分布的200个ROI
        if n_rois <= 250:
            # 如果接近200，选择前200个
            aligned = timeseries[:, :target_rois]
        else:
            # 如果远大于200，均匀采样
            indices = np.linspace(0, n_rois - 1, target_rois, dtype=int)
            aligned = timeseries[:, indices]
        
        return aligned


def compute_pcc(timeseries: np.ndarray) -> np.ndarray:
    """
    计算Pearson相关系数矩阵并提取上三角向量
    
    Args:
        timeseries: ndarray, shape (n_timepoints, n_rois)
                   每行是一个时间点，每列是一个ROI
                   注意：函数内部会将输入转置为 (n_rois, n_timepoints) 以便计算ROI间的相关系数
    
    Returns:
        pcc_upper: ndarray, shape (n_rois * (n_rois - 1) // 2,)
    """
    # 输入假设是 (n_timepoints, n_rois)
    # 需要转置为 (n_rois, n_timepoints)，因为 np.corrcoef 按行计算相关系数
    # 我们希望每行是一个ROI的时间序列，以便计算ROI之间的相关系数
    
    # 总是转置：从 (n_timepoints, n_rois) 到 (n_rois, n_timepoints)
    timeseries_t = timeseries.T  # (n_rois, n_timepoints)
    
    # 计算相关系数矩阵: corrcoef按行计算，输出(n_rois, n_rois)
    # 每个元素 corr[i,j] 表示 ROI i 和 ROI j 之间的相关系数
    pcc_matrix = np.corrcoef(timeseries_t)
    
    # 处理NaN值
    pcc_matrix = np.nan_to_num(pcc_matrix, nan=0.0)
    
    # 提取上三角（不包含对角线）
    n_rois = pcc_matrix.shape[0]
    upper_indices = np.triu_indices(n_rois, k=1)
    pcc_upper = pcc_matrix[upper_indices]
    
    return pcc_upper


def apply_sliding_window(
    timeseries: np.ndarray,
    window_size: int = 32,
    stride: int = 16
) -> Dict:
    """
    应用滑动窗口切分
    
    Args:
        timeseries: ndarray, shape (n_timepoints, n_rois)
        window_size: 窗口大小
        stride: 滑动步长
    
    Returns:
        windowed_data: dict包含窗口化数据
    """
    n_timepoints, n_rois = timeseries.shape
    n_windows = (n_timepoints - window_size) // stride + 1
    
    if n_windows <= 0:
        raise ValueError(f"时间点数量({n_timepoints})不足以创建窗口(大小{window_size})")
    
    all_windows = []
    all_pcc_vectors = []
    window_indices = []
    
    for win_idx in range(n_windows):
        start = win_idx * stride
        end = start + window_size
        
        window = timeseries[start:end]  # (window_size, n_rois)
        
        # 从窗口计算PCC
        window_pcc = compute_pcc(window)
        
        all_windows.append(window)
        all_pcc_vectors.append(window_pcc)
        window_indices.append((start, end))
    
    windowed_data = {
        'timeseries': np.array(all_windows, dtype=np.float32),  # (n_windows, window_size, n_rois)
        'pcc_vectors': np.array(all_pcc_vectors, dtype=np.float32),  # (n_windows, pcc_dim)
        'window_indices': window_indices,
        'window_info': {
            'window_size': window_size,
            'stride': stride,
            'n_windows': n_windows,
            'n_timepoints': n_timepoints,
            'n_rois': n_rois
        }
    }
    
    return windowed_data


def process_input_data(
    data: Union[str, dict, np.ndarray],  # 移除了 pd.DataFrame 类型提示，使用延迟导入
    input_format: str = 'auto',
    window_size: int = 32,
    stride: int = 16,
    align_cc200: bool = True
) -> Dict:
    """
    处理用户输入的数据（JSON/CSV/ndarray）
    
    Args:
        data: 输入数据（JSON字符串、字典、CSV路径、ndarray、DataFrame）
        input_format: 输入格式 ('json', 'csv', 'auto')
        window_size: 滑动窗口大小
        stride: 滑动窗口步长
        align_cc200: 是否对齐到CC200
    
    Returns:
        processed_data: dict包含处理后的数据
    """
    # 1. 加载数据
    if input_format == 'auto':
        if isinstance(data, str):
            if data.endswith('.csv'):
                input_format = 'csv'
            elif data.endswith('.json') or data.strip().startswith('{') or data.strip().startswith('['):
                input_format = 'json'
            else:
                raise ValueError(f"无法自动识别格式: {data[:100]}")
        elif isinstance(data, dict) or isinstance(data, list):
            input_format = 'json'
        elif isinstance(data, np.ndarray):
            input_format = 'array'
        elif hasattr(data, 'values') and hasattr(data, 'columns'):  # DataFrame-like
            input_format = 'dataframe'
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    if input_format == 'csv':
        timeseries = load_fmri_data_from_csv(data)
    elif input_format == 'json':
        timeseries = load_fmri_data_from_json(data)
    elif input_format == 'array':
        timeseries = np.array(data, dtype=np.float32)
        if timeseries.ndim != 2:
            raise ValueError(f"数组必须是2D，当前形状: {timeseries.shape}")
        if timeseries.shape[0] < timeseries.shape[1]:
            timeseries = timeseries.T
    elif input_format == 'dataframe':
        # 延迟导入 pandas
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            timeseries = data.values.astype(np.float32)
        else:
            timeseries = np.array(data).astype(np.float32)
        if timeseries.shape[0] < timeseries.shape[1]:
            timeseries = timeseries.T
    else:
        raise ValueError(f"不支持的格式: {input_format}")
    
    # 2. 对齐到CC200
    if align_cc200:
        timeseries = align_to_cc200(timeseries)
    
    # 3. 应用滑动窗口
    windowed_data = apply_sliding_window(timeseries, window_size, stride)
    
    processed_data = {
        'original_timeseries': timeseries,
        'windowed_data': windowed_data,
        'n_rois': timeseries.shape[1],
        'n_timepoints': timeseries.shape[0]
    }
    
    return processed_data