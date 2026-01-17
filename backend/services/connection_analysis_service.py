"""
异常连接分析服务
分析CC200脑区的异常连接模式
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def pcc_vector_to_matrix(pcc_vector: np.ndarray, n_rois: int = 200) -> np.ndarray:
    """
    将PCC向量（上三角）转换为200x200的连接矩阵
    
    Args:
        pcc_vector: ndarray, shape (pcc_dim,) 或 (batch, pcc_dim)
        n_rois: ROI数量（默认200）
    
    Returns:
        pcc_matrix: ndarray, shape (n_rois, n_rois) 或 (batch, n_rois, n_rois)
    """
    if pcc_vector.ndim == 1:
        matrix = np.zeros((n_rois, n_rois))
        upper_indices = np.triu_indices(n_rois, k=1)
        matrix[upper_indices] = pcc_vector
        # 对称填充
        matrix = matrix + matrix.T
        # 对角线设为1（自身连接）
        np.fill_diagonal(matrix, 1.0)
        return matrix
    else:
        batch_size = pcc_vector.shape[0]
        matrices = np.zeros((batch_size, n_rois, n_rois))
        upper_indices = np.triu_indices(n_rois, k=1)
        for i in range(batch_size):
            matrices[i][upper_indices] = pcc_vector[i]
            matrices[i] = matrices[i] + matrices[i].T
            np.fill_diagonal(matrices[i], 1.0)
        return matrices


class ConnectionAnalysisService:
    """异常连接分析服务类"""
    
    def __init__(self, n_rois: int = 200):
        """
        初始化分析服务
        
        Args:
            n_rois: ROI数量（CC200为200）
        """
        self.n_rois = n_rois
        self.pcc_dim = n_rois * (n_rois - 1) // 2
    
    def analyze_abnormal_connections(
        self,
        pcc_vectors: np.ndarray,
        predictions: Dict,
        threshold: float = 0.5,
        top_k: int = 50
    ) -> Dict:
        """
        分析异常连接模式
        
        Args:
            pcc_vectors: PCC向量数组，shape (n_windows, pcc_dim)
            predictions: 预测结果dict
            threshold: ASD概率阈值（高于此值认为是ASD窗口）
            top_k: 返回top-k异常连接
        
        Returns:
            analysis: dict包含异常连接分析结果
        """
        n_windows = len(pcc_vectors)
        
        # 根据预测概率分离ASD和TC窗口
        window_probs = [w['asd_probability'] for w in predictions['window_predictions']]
        asd_mask = np.array(window_probs) >= threshold
        tc_mask = np.array(window_probs) < threshold
        
        asd_indices = np.where(asd_mask)[0]
        tc_indices = np.where(tc_mask)[0]
        
        if len(asd_indices) == 0:
            print("Warning: 没有检测到ASD窗口（根据阈值）")
            # 使用top-k概率的窗口作为ASD窗口
            top_indices = np.argsort(window_probs)[-min(n_windows // 2, 10):]
            asd_indices = top_indices
            tc_indices = np.setdiff1d(np.arange(n_windows), asd_indices)
        
        if len(tc_indices) == 0:
            print("Warning: 没有检测到TC窗口（根据阈值）")
            # 使用bottom-k概率的窗口作为TC窗口
            bottom_indices = np.argsort(window_probs)[:min(n_windows // 2, 10)]
            tc_indices = bottom_indices
            asd_indices = np.setdiff1d(np.arange(n_windows), tc_indices)
        
        # 计算ASD和TC的平均PCC向量
        asd_pcc = pcc_vectors[asd_indices]
        tc_pcc = pcc_vectors[tc_indices]
        
        asd_mean_pcc = np.mean(asd_pcc, axis=0)
        tc_mean_pcc = np.mean(tc_pcc, axis=0)
        
        # 计算差异
        pcc_diff = asd_mean_pcc - tc_mean_pcc
        
        # 转换为连接矩阵
        asd_mean_matrix = pcc_vector_to_matrix(asd_mean_pcc, self.n_rois)
        tc_mean_matrix = pcc_vector_to_matrix(tc_mean_pcc, self.n_rois)
        diff_matrix = pcc_vector_to_matrix(pcc_diff, self.n_rois)
        
        # 找出差异最大的连接（绝对值）
        diff_abs = np.abs(diff_matrix)
        upper_indices = np.triu_indices(self.n_rois, k=1)
        
        # 找出top-k差异最大的连接
        diff_upper = diff_abs[upper_indices]
        top_k_indices = np.argsort(diff_upper)[-top_k:]
        
        # 转换为(i, j)坐标
        top_k_connections = [
            {
                'roi_i': int(upper_indices[0][idx]),
                'roi_j': int(upper_indices[1][idx]),
                'difference': float(diff_upper[idx]),
                'asd_connection_strength': float(asd_mean_matrix[upper_indices[0][idx], upper_indices[1][idx]]),
                'tc_connection_strength': float(tc_mean_matrix[upper_indices[0][idx], upper_indices[1][idx]])
            }
            for idx in reversed(top_k_indices)  # 从大到小排序
        ]
        
        # 计算连接强度统计
        connection_stats = {
            'asd': {
                'mean_connection_strength': float(np.mean(np.abs(asd_mean_matrix))),
                'std_connection_strength': float(np.std(asd_mean_matrix)),
                'n_windows': int(len(asd_indices))
            },
            'tc': {
                'mean_connection_strength': float(np.mean(np.abs(tc_mean_matrix))),
                'std_connection_strength': float(np.std(tc_mean_matrix)),
                'n_windows': int(len(tc_indices))
            }
        }
        
        analysis = {
            'abnormal_connections': top_k_connections,
            'connection_statistics': connection_stats,
            'summary': {
                'n_rois': self.n_rois,
                'n_abnormal_connections_identified': len(top_k_connections),
                'threshold_used': threshold,
                'asd_windows_count': int(len(asd_indices)),
                'tc_windows_count': int(len(tc_indices))
            }
        }
        
        return analysis
    
    def get_cc200_roi_labels(self) -> List[Dict]:
        """
        获取CC200的ROI标签（如果有的话）
        
        Returns:
            roi_labels: ROI标签列表
        """
        # 这里可以返回CC200的ROI标签，如果有的话
        # 目前返回索引
        return [
            {'roi_index': i, 'roi_name': f'ROI_{i:03d}'}
            for i in range(self.n_rois)
        ]


# 导出pcc_vector_to_matrix函数供外部使用
__all__ = ['ConnectionAnalysisService', 'pcc_vector_to_matrix']