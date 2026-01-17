"""
诊断预测流程
检查模型是否正确加载、是否使用正确的融合方式、参数是否正确
"""

import os
import sys
import json
import torch
import numpy as np

# 添加API目录到路径
API_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, API_ROOT)

from models.dual_stream import DualStreamModel
from services.prediction_service import PredictionService

def diagnose_model_loading():
    """诊断模型加载"""
    print("=" * 80)
    print("诊断 1: 模型加载检查")
    print("=" * 80)
    
    checkpoint_path = os.path.join(API_ROOT, 'checkpoints', 'best_model.pt')
    results_json_path = os.path.join(API_ROOT, 'checkpoints', 'results.json')
    
    # 1. 检查文件是否存在
    print(f"\n1. 检查文件存在性:")
    print(f"   - checkpoint: {checkpoint_path}")
    print(f"     存在: {os.path.exists(checkpoint_path)}")
    print(f"   - results.json: {results_json_path}")
    print(f"     存在: {os.path.exists(results_json_path)}")
    
    if not os.path.exists(checkpoint_path):
        print("   [ERROR] checkpoint文件不存在！")
        return False
    
    if not os.path.exists(results_json_path):
        print("   [ERROR] results.json文件不存在！")
        return False
    
    # 2. 检查results.json中的最佳epoch
    print(f"\n2. 检查results.json中的最佳参数:")
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    best_epoch = results.get('best_epoch', None)
    best_val_auc = results.get('best_val_auc', None)
    accuracy = results.get('accuracy', None)
    fusion_type = results.get('config', {}).get('fusion', {}).get('type', None)
    
    print(f"   - best_epoch: {best_epoch}")
    print(f"   - best_val_auc: {best_val_auc:.4f}" if best_val_auc else "   - best_val_auc: 未找到")
    print(f"   - accuracy: {accuracy:.4f}" if accuracy else "   - accuracy: 未找到")
    print(f"   - fusion_type: {fusion_type}")
    
    if fusion_type != 'cross_attention':
        print(f"   [WARN] 警告: fusion_type不是cross_attention，而是{fusion_type}")
    
    # 3. 检查checkpoint中的epoch信息
    print(f"\n3. 检查checkpoint中的epoch信息:")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    checkpoint_epoch = checkpoint.get('epoch', None)
    checkpoint_val_auc = checkpoint.get('val_auc', None)
    
    print(f"   - checkpoint epoch: {checkpoint_epoch}")
    print(f"   - checkpoint val_auc: {checkpoint_val_auc:.4f}" if checkpoint_val_auc else "   - checkpoint val_auc: 未找到")
    
    if checkpoint_epoch is not None and best_epoch is not None:
        if checkpoint_epoch == best_epoch:
            print(f"   [OK] checkpoint epoch ({checkpoint_epoch}) 匹配最佳epoch ({best_epoch})")
        else:
            print(f"   [WARN] 警告: checkpoint epoch ({checkpoint_epoch}) 不匹配最佳epoch ({best_epoch})")
    
    # 4. 检查模型结构
    print(f"\n4. 检查模型结构:")
    config = results.get('config', {})
    model_config = config.get('model', {})
    fusion_config = config.get('fusion', {})
    
    tst1_config = model_config.get('tst1', {})
    tst2_config = model_config.get('tst2', {})
    fusion_type = fusion_config.get('type', 'cross_attention')
    fusion_kwargs = fusion_config.get(fusion_type, {})
    
    print(f"   - TST1配置: {tst1_config}")
    print(f"   - TST2配置: {tst2_config}")
    print(f"   - Fusion类型: {fusion_type}")
    print(f"   - Fusion配置: {fusion_kwargs}")
    
    # 5. 创建模型并检查
    print(f"\n5. 创建模型并检查结构:")
    model = DualStreamModel(
        tst1_config=tst1_config,
        tst2_config=tst2_config,
        fusion_type=fusion_type,
        fusion_config=fusion_kwargs,
        num_classes=2,
        dropout=config.get('finetune', {}).get('classifier', {}).get('dropout', 0.3)
    )
    
    print(f"   - 模型类型: {type(model).__name__}")
    print(f"   - Fusion模块类型: {type(model.fusion).__name__}")
    print(f"   - 是否使用CrossAttention: {type(model.fusion).__name__ == 'CrossAttentionFusion'}")
    
    if type(model.fusion).__name__ != 'CrossAttentionFusion':
        print(f"   [WARN] 警告: 模型使用的融合方式不是CrossAttentionFusion！")
    
    # 6. 检查checkpoint中的键
    print(f"\n6. 检查checkpoint中的键:")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    fusion_keys = [k for k in state_dict.keys() if 'fusion' in k.lower()]
    tst1_keys = [k for k in state_dict.keys() if 'tst1' in k.lower() or 'transformer_ts' in k.lower()]
    tst2_keys = [k for k in state_dict.keys() if 'tst2' in k.lower() or 'transformer_fc' in k.lower()]
    
    print(f"   - Fusion相关键数量: {len(fusion_keys)}")
    print(f"   - TST1相关键数量: {len(tst1_keys)}")
    print(f"   - TST2相关键数量: {len(tst2_keys)}")
    
    if fusion_keys:
        print(f"   - Fusion键示例: {fusion_keys[:3]}")
        # 检查是否有cross_attention相关的键
        cross_attn_keys = [k for k in fusion_keys if 'cross' in k.lower() or 'attn' in k.lower()]
        print(f"   - Cross-Attention相关键数量: {len(cross_attn_keys)}")
        if cross_attn_keys:
            print(f"   - Cross-Attention键示例: {cross_attn_keys[:3]}")
    
    return True


def diagnose_prediction_flow():
    """诊断预测流程"""
    print("\n" + "=" * 80)
    print("诊断 2: 预测流程检查")
    print("=" * 80)
    
    checkpoint_path = os.path.join(API_ROOT, 'checkpoints', 'best_model.pt')
    results_json_path = os.path.join(API_ROOT, 'checkpoints', 'results.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # 创建预测服务
        print(f"\n1. 创建PredictionService:")
        pred_service = PredictionService(checkpoint_path, results_json_path, device)
        print(f"   [OK] PredictionService创建成功")
        
        # 检查模型
        print(f"\n2. 检查模型:")
        model = pred_service.model
        print(f"   - 模型类型: {type(model).__name__}")
        print(f"   - Fusion类型: {type(model.fusion).__name__}")
        print(f"   - 设备: {next(model.parameters()).device}")
        print(f"   - 训练模式: {'训练' if model.training else '评估'}")
        
        # 3. 测试预测
        print(f"\n3. 测试预测流程:")
        # 创建模拟数据
        n_windows = 5
        window_size = 32
        n_rois = 200
        pcc_dim = 19900
        
        timeseries = np.random.randn(n_windows, window_size, n_rois)
        pcc_vectors = np.random.randn(n_windows, pcc_dim)
        
        windowed_data = {
            'timeseries': timeseries,
            'pcc_vectors': pcc_vectors
        }
        
        print(f"   - 输入数据形状:")
        print(f"     * timeseries: {timeseries.shape}")
        print(f"     * pcc_vectors: {pcc_vectors.shape}")
        
        # 执行预测
        predictions = pred_service.predict(windowed_data, batch_size=2, return_attention=False)
        
        print(f"   [OK] 预测成功")
        print(f"   - 窗口数量: {len(predictions['window_predictions'])}")
        print(f"   - 平均ASD概率: {predictions['summary']['mean_asd_probability']:.4f}")
        
        # 4. 检查是否使用了dual_stream模型
        print(f"\n4. 检查模型使用:")
        print(f"   - 是否使用DualStreamModel: {isinstance(model, DualStreamModel)}")
        print(f"   - 是否使用CrossAttentionFusion: {type(model.fusion).__name__ == 'CrossAttentionFusion'}")
        
        # 5. 检查前向传播路径
        print(f"\n5. 检查前向传播路径:")
        # 手动执行一次前向传播
        model.eval()
        with torch.no_grad():
            batch_ts = torch.FloatTensor(timeseries[:2]).to(device)
            batch_pcc = torch.FloatTensor(pcc_vectors[:2]).to(device)
            
            # 标准化
            batch_ts = (batch_ts - batch_ts.mean()) / (batch_ts.std() + 1e-8)
            batch_pcc = (batch_pcc - batch_pcc.mean()) / (batch_pcc.std() + 1e-8)
            
            # 前向传播
            logits = model(batch_ts, batch_pcc)
            print(f"   - 输入形状: timeseries={batch_ts.shape}, pcc={batch_pcc.shape}")
            print(f"   - 输出logits形状: {logits.shape}")
            print(f"   - 输出logits值: {logits.cpu().numpy()}")
            
            # 检查中间特征
            logits, fused, h_ts, h_fc = model(batch_ts, batch_pcc, return_features=True)
            print(f"   - TST1特征形状: {h_ts.shape}")
            print(f"   - TST2特征形状: {h_fc.shape}")
            print(f"   - 融合特征形状: {fused.shape}")
            print(f"   [OK] 前向传播路径正常")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_parameter_loading():
    """诊断参数加载"""
    print("\n" + "=" * 80)
    print("诊断 3: 参数加载检查")
    print("=" * 80)
    
    checkpoint_path = os.path.join(API_ROOT, 'checkpoints', 'best_model.pt')
    results_json_path = os.path.join(API_ROOT, 'checkpoints', 'results.json')
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载配置
        with open(results_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        config = results.get('config', {})
        model_config = config.get('model', {})
        fusion_config = config.get('fusion', {})
        
        tst1_config = model_config.get('tst1', {})
        tst2_config = model_config.get('tst2', {})
        fusion_type = fusion_config.get('type', 'cross_attention')
        fusion_kwargs = fusion_config.get(fusion_type, {})
        
        # 创建模型
        model = DualStreamModel(
            tst1_config=tst1_config,
            tst2_config=tst2_config,
            fusion_type=fusion_type,
            fusion_config=fusion_kwargs,
            num_classes=2,
            dropout=config.get('finetune', {}).get('classifier', {}).get('dropout', 0.3)
        )
        
        # 转换键名
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('tst1.'):
                new_key = key.replace('tst1.', 'transformer_ts.', 1)
            elif key.startswith('tst2.'):
                new_key = key.replace('tst2.', 'transformer_fc.', 1)
            new_state_dict[new_key] = value
        
        # 检查参数匹配
        model_state_dict = model.state_dict()
        
        print(f"\n1. 参数匹配检查:")
        matched_keys = []
        mismatched_keys = []
        missing_keys = []
        
        for key in model_state_dict.keys():
            if key in new_state_dict:
                if model_state_dict[key].shape == new_state_dict[key].shape:
                    matched_keys.append(key)
                else:
                    mismatched_keys.append(key)
            else:
                missing_keys.append(key)
        
        print(f"   - 匹配的参数: {len(matched_keys)}")
        print(f"   - 尺寸不匹配的参数: {len(mismatched_keys)}")
        print(f"   - 缺失的参数: {len(missing_keys)}")
        
        # 检查关键参数
        print(f"\n2. 关键参数检查:")
        critical_keys = [
            'transformer_ts.input_embedding.weight',
            'transformer_fc.input_embedding.weight',
            'fusion.proj_ts.weight',
            'fusion.proj_fc.weight',
        ]
        
        if fusion_type == 'cross_attention':
            critical_keys.extend([
                'fusion.cross_attn_ts2fc.in_proj_weight',
                'fusion.cross_attn_fc2ts.in_proj_weight',
            ])
        
        for key in critical_keys:
            if key in matched_keys:
                print(f"   [OK] {key}: 已加载")
            elif key in mismatched_keys:
                print(f"   [WARN] {key}: 尺寸不匹配")
            else:
                print(f"   [ERROR] {key}: 缺失")
        
        # 检查融合模块参数
        print(f"\n3. 融合模块参数检查:")
        fusion_keys = [k for k in matched_keys if k.startswith('fusion.')]
        print(f"   - 融合模块已加载参数数量: {len(fusion_keys)}")
        if fusion_keys:
            print(f"   - 融合模块参数示例: {fusion_keys[:5]}")
        
        return len(matched_keys) > 0
        
    except Exception as e:
        print(f"   [ERROR] 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("预测流程诊断工具")
    print("=" * 80)
    
    results = []
    
    # 诊断1: 模型加载
    result1 = diagnose_model_loading()
    results.append(("模型加载", result1))
    
    # 诊断2: 预测流程
    result2 = diagnose_prediction_flow()
    results.append(("预测流程", result2))
    
    # 诊断3: 参数加载
    result3 = diagnose_parameter_loading()
    results.append(("参数加载", result3))
    
    # 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    
    for name, result in results:
        status = "[OK] 通过" if result else "[ERROR] 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n[OK] 所有诊断通过！模型应该能正常工作。")
    else:
        print("\n[WARN] 部分诊断失败，请检查上述问题。")
    
    return all_passed


if __name__ == '__main__':
    main()
