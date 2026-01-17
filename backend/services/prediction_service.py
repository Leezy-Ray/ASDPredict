"""
预测服务
加载模型并进行预测
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

# 添加API目录到路径（用于相对导入）
API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_ROOT)

from models.dual_stream import DualStreamModel


class PredictionService:
    """预测服务类"""
    
    def __init__(
        self,
        checkpoint_path: str,
        results_json_path: str,
        device: str = 'cuda'
    ):
        """
        初始化预测服务
        
        Args:
            checkpoint_path: 模型checkpoint路径
            results_json_path: results.json路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.results_json_path = results_json_path
        self.model = None
        self.config = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"Loading model from {self.checkpoint_path}...")
        
        # 加载配置
        with open(self.results_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.config = results.get('config', {})
        model_config = self.config.get('model', {})
        fusion_config = self.config.get('fusion', {})
        
        tst1_config = model_config.get('tst1', {})
        tst2_config = model_config.get('tst2', {})
        fusion_type = fusion_config.get('type', 'cross_attention')
        fusion_kwargs = fusion_config.get(fusion_type, {})
        
        # 获取classifier配置
        classifier_config = self.config.get('finetune', {}).get('classifier', {})
        classifier_hidden_dims = classifier_config.get('hidden_dims', None)
        classifier_dropout = classifier_config.get('dropout', 0.3)
        
        # 创建模型
        self.model = DualStreamModel(
            tst1_config=tst1_config,
            tst2_config=tst2_config,
            fusion_type=fusion_type,
            fusion_config=fusion_kwargs,
            num_classes=2,
            dropout=classifier_dropout,
            classifier_hidden_dims=classifier_hidden_dims
        )
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 转换键名：tst1 -> transformer_ts, tst2 -> transformer_fc
        new_state_dict = {}
        key_transforms = 0
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('tst1.'):
                new_key = key.replace('tst1.', 'transformer_ts.', 1)
                key_transforms += 1
            elif key.startswith('tst2.'):
                new_key = key.replace('tst2.', 'transformer_fc.', 1)
                key_transforms += 1
            new_state_dict[new_key] = value
        
        print(f"Transformed {key_transforms} keys from checkpoint (tst1/tst2 -> transformer_ts/transformer_fc)")
        
        # 获取模型当前的state_dict以检查尺寸匹配
        model_state_dict = self.model.state_dict()
        
        # 过滤掉尺寸不匹配的键（特别是classifier层）
        filtered_state_dict = {}
        size_mismatches = []
        for key, value in new_state_dict.items():
            if key in model_state_dict:
                # 检查尺寸是否匹配
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    size_mismatches.append(f"{key}: checkpoint {list(value.shape)} vs model {list(model_state_dict[key].shape)}")
            else:
                # 键不存在，跳过
                pass
        
        if size_mismatches:
            print(f"Warning: Skipping {len(size_mismatches)} keys with size mismatches:")
            for mismatch in size_mismatches[:5]:  # 只显示前5个
                print(f"  - {mismatch}")
            if len(size_mismatches) > 5:
                print(f"  ... and {len(size_mismatches) - 5} more")
        
        # 统计转换后的键
        transformer_ts_keys = [k for k in filtered_state_dict.keys() if k.startswith('transformer_ts.')]
        transformer_fc_keys = [k for k in filtered_state_dict.keys() if k.startswith('transformer_fc.')]
        fusion_keys = [k for k in filtered_state_dict.keys() if k.startswith('fusion.')]
        classifier_keys = [k for k in filtered_state_dict.keys() if k.startswith('classifier.')]
        
        print(f"State dict summary:")
        print(f"  - transformer_ts keys: {len(transformer_ts_keys)}")
        print(f"  - transformer_fc keys: {len(transformer_fc_keys)}")
        print(f"  - fusion keys: {len(fusion_keys)}")
        print(f"  - classifier keys: {len(classifier_keys)}")
        print(f"  - Total filtered keys: {len(filtered_state_dict)}")
        
        # 验证关键参数是否存在
        critical_keys = [
            'transformer_fc.input_embedding.weight',
            'transformer_ts.input_embedding.weight'
        ]
        missing_critical = [k for k in critical_keys if k not in filtered_state_dict]
        if missing_critical:
            print(f"ERROR: Critical keys missing from filtered state dict: {missing_critical}")
        
        # 加载过滤后的权重（strict=False）
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # 验证参数是否真的被加载了
        loaded_transformer_ts = len([k for k in filtered_state_dict.keys() if k.startswith('transformer_ts.') and k not in missing_keys])
        loaded_transformer_fc = len([k for k in filtered_state_dict.keys() if k.startswith('transformer_fc.') and k not in missing_keys])
        print(f"Successfully loaded: transformer_ts={loaded_transformer_ts} keys, transformer_fc={loaded_transformer_fc} keys")
        
        # 验证 input_embedding 维度
        if hasattr(self.model.transformer_fc, 'input_embedding'):
            fc_input_dim = self.model.transformer_fc.input_embedding.in_features
            fc_output_dim = self.model.transformer_fc.input_embedding.out_features
            print(f"transformer_fc.input_embedding: {fc_input_dim} -> {fc_output_dim}")
        
        if missing_keys:
            # 过滤掉尺寸不匹配的层
            actual_missing = [k for k in missing_keys if k not in ['classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']]
            if actual_missing:
                print(f"Warning: Missing keys (will use random initialization): {actual_missing[:10]}...")
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys (will be ignored): {unexpected_keys[:10]}...")
        
        # 检查classifier尺寸不匹配
        classifier_mismatch = [k for k in missing_keys if k.startswith('classifier.')]
        if classifier_mismatch:
            print(f"Warning: Classifier size mismatch, using model's default classifier. Mismatched keys: {classifier_mismatch}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(
        self,
        windowed_data: Dict,
        batch_size: int = 32,
        return_attention: bool = False
    ) -> Dict:
        """
        对窗口化数据进行预测
        
        Args:
            windowed_data: 窗口化数据dict，包含timeseries和pcc_vectors
            batch_size: 批次大小
            return_attention: 是否返回注意力权重
        
        Returns:
            predictions: dict包含预测结果
        """
        timeseries = windowed_data['timeseries']  # (n_windows, window_size, n_rois)
        pcc_vectors = windowed_data['pcc_vectors']  # (n_windows, pcc_dim)
        
        # 转换为tensor
        timeseries_tensor = torch.FloatTensor(timeseries)
        pcc_vectors_tensor = torch.FloatTensor(pcc_vectors)
        
        n_windows = len(timeseries)
        all_probs = []
        all_preds = []
        all_logits = []
        all_attention_weights = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                end_idx = min(i + batch_size, n_windows)
                batch_ts = timeseries_tensor[i:end_idx].to(self.device)
                batch_pcc = pcc_vectors_tensor[i:end_idx].to(self.device)
                
                # 标准化（使用全局统计量）
                # 这里简化处理，实际应该使用训练时的统计量
                batch_ts = (batch_ts - batch_ts.mean()) / (batch_ts.std() + 1e-8)
                batch_pcc = (batch_pcc - batch_pcc.mean()) / (batch_pcc.std() + 1e-8)
                
                # 预测
                if return_attention:
                    result = self.model(batch_ts, batch_pcc, return_attention=True)
                    if isinstance(result, tuple):
                        logits = result[0]
                        if len(result) > 1 and isinstance(result[-1], dict):
                            all_attention_weights.append(result[-1])
                        else:
                            all_attention_weights.append(None)
                    else:
                        logits = result
                        all_attention_weights.append(None)
                else:
                    logits = self.model(batch_ts, batch_pcc)
                    all_attention_weights.append(None)
                
                probs = F.softmax(logits, dim=1)
                
                all_logits.extend(logits.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # ASD概率
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        predictions = {
            'window_predictions': [
                {
                    'window_index': i,
                    'asd_probability': float(all_probs[i]),
                    'prediction': int(all_preds[i]),
                    'logits': all_logits[i].tolist()
                }
                for i in range(n_windows)
            ],
            'summary': {
                'total_windows': n_windows,
                'mean_asd_probability': float(np.mean(all_probs)),
                'std_asd_probability': float(np.std(all_probs)),
                'predicted_asd_windows': int(np.sum(all_preds)),
                'predicted_tc_windows': int(n_windows - np.sum(all_preds))
            }
        }
        
        if return_attention and any(aw is not None for aw in all_attention_weights):
            predictions['attention_weights'] = all_attention_weights
        
        return predictions