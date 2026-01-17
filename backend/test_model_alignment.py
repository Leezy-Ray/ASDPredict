"""
测试脚本：验证模型参数对齐和数据流
"""

import os
import sys
import json
import torch
import numpy as np

# 添加路径（与app.py相同的设置）
API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(__file__)) == 'backend' else os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_ROOT)

# 检查transformer_ts.py和transformer_fc.py是否存在
import os
transformer_ts_path = os.path.join(BACKEND_ROOT, 'models', 'transformer_ts.py')
transformer_fc_path = os.path.join(BACKEND_ROOT, 'models', 'transformer_fc.py')
tst1_path = os.path.join(BACKEND_ROOT, 'models', 'tst1.py')
tst2_path = os.path.join(BACKEND_ROOT, 'models', 'tst2.py')

# 如果不存在transformer_ts.py/transformer_fc.py，创建符号链接或复制
if not os.path.exists(transformer_ts_path) and os.path.exists(tst1_path):
    import shutil
    print(f"创建 transformer_ts.py -> tst1.py 的副本...")
    shutil.copy(tst1_path, transformer_ts_path)
if not os.path.exists(transformer_fc_path) and os.path.exists(tst2_path):
    import shutil
    print(f"创建 transformer_fc.py -> tst2.py 的副本...")
    shutil.copy(tst2_path, transformer_fc_path)

# 现在可以导入（使用与app.py相同的导入方式）
from models.dual_stream import DualStreamModel
from utils.data_processor import process_input_data, compute_pcc

# #region agent log
log_path = r'd:\workplace\ASDModelPredict\.cursor\debug.log'
def log_debug(location, message, data):
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"location": location, "message": message, "data": data, "timestamp": int(__import__('time').time()*1000), "sessionId": "debug-session", "runId": "test-alignment", "hypothesisId": "A"}) + '\n')
    except:
        pass
# #endregion

def test_data_flow():
    """测试数据流"""
    print("=" * 60)
    print("测试 1: 数据流验证")
    print("=" * 60)
    
    # 模拟输入：100个时间点，200个脑区
    n_timepoints = 100
    n_rois = 200
    window_size = 32
    stride = 16
    
    # 生成模拟数据
    sample_data = np.random.randn(n_timepoints, n_rois).astype(np.float32)
    print(f"输入数据形状: {sample_data.shape} (时间点, 脑区)")
    log_debug("test_model_alignment.py:test_data_flow", "Input data shape", {"shape": list(sample_data.shape)})
    
    # 处理数据
    processed_data = process_input_data(
        data=sample_data,
        input_format='array',
        window_size=window_size,
        stride=stride,
        align_cc200=True
    )
    
    windowed_data = processed_data['windowed_data']
    timeseries_windows = windowed_data['timeseries']  # (n_windows, window_size, n_rois)
    pcc_vectors = windowed_data['pcc_vectors']  # (n_windows, pcc_dim)
    
    n_windows = len(timeseries_windows)
    print(f"\n窗口化结果:")
    print(f"  - 窗口数量: {n_windows}")
    print(f"  - 每个窗口时间序列形状: {timeseries_windows[0].shape} (窗口大小, 脑区数)")
    print(f"  - PCC向量形状: {pcc_vectors[0].shape} (应该是 {200*199//2} = {200*199//2})")
    
    expected_pcc_dim = n_rois * (n_rois - 1) // 2
    actual_pcc_dim = pcc_vectors[0].shape[0]
    
    log_debug("test_model_alignment.py:test_data_flow", "Windowed data shapes", {
        "n_windows": n_windows,
        "timeseries_shape": list(timeseries_windows[0].shape),
        "pcc_vector_shape": list(pcc_vectors[0].shape),
        "expected_pcc_dim": expected_pcc_dim,
        "actual_pcc_dim": actual_pcc_dim
    })
    
    if actual_pcc_dim != expected_pcc_dim:
        print(f"  [FAIL] PCC维度不匹配！期望 {expected_pcc_dim}，实际 {actual_pcc_dim}")
        print(f"  原因分析: 496 = 32*31/2，说明计算PCC时使用了32个ROI而不是200个")
        return False
    else:
        print(f"  [PASS] PCC维度正确: {actual_pcc_dim}")
    
    return True


def test_model_structure():
    """测试模型结构"""
    print("\n" + "=" * 60)
    print("测试 2: 模型结构验证")
    print("=" * 60)
    
    # 加载配置
    BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
    results_json_path = os.path.join(BACKEND_ROOT, 'checkpoints', 'results.json')
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    config = results.get('config', {})
    model_config = config.get('model', {})
    fusion_config = config.get('fusion', {})
    
    tst1_config = model_config.get('tst1', {})
    tst2_config = model_config.get('tst2', {})
    fusion_type = fusion_config.get('type', 'cross_attention')
    fusion_kwargs = fusion_config.get(fusion_type, {})
    
    print(f"TST1 配置: {tst1_config}")
    print(f"TST2 配置: {tst2_config}")
    print(f"融合类型: {fusion_type}")
    
    # 创建模型
    model = DualStreamModel(
        tst1_config=tst1_config,
        tst2_config=tst2_config,
        fusion_type=fusion_type,
        fusion_config=fusion_kwargs,
        num_classes=2,
        dropout=config.get('finetune', {}).get('classifier', {}).get('dropout', 0.3)
    )
    
    # 检查关键参数
    print(f"\n模型结构:")
    print(f"  - TST1 emb_dim: {model.transformer_ts.emb_dim}")
    print(f"  - TST2 d_model: {model.transformer_fc.d_model}")
    print(f"  - TST2 pcc_dim: {model.transformer_fc.pcc_dim}")
    print(f"  - Fusion output_dim: {model.fusion.output_dim}")
    print(f"  - Classifier 第一层输入: {model.classifier[0].in_features}")
    
    log_debug("test_model_alignment.py:test_model_structure", "Model structure", {
        "tst1_emb_dim": model.transformer_ts.emb_dim,
        "tst2_d_model": model.transformer_fc.d_model,
        "tst2_pcc_dim": model.transformer_fc.pcc_dim,
        "fusion_output_dim": model.fusion.output_dim,
        "classifier_input_dim": model.classifier[0].in_features
    })
    
    # 验证TST2的input_embedding维度
    tst2_input_dim = model.transformer_fc.input_embedding.in_features
    expected_pcc_dim = 200 * 199 // 2  # 19900
    
    if tst2_input_dim != expected_pcc_dim:
        print(f"  [FAIL] TST2 input_embedding维度不匹配！期望 {expected_pcc_dim}，实际 {tst2_input_dim}")
        return False
    else:
        print(f"  [PASS] TST2 input_embedding维度正确: {tst2_input_dim}")
    
    return True, model


def test_checkpoint_loading(model):
    """测试checkpoint加载"""
    print("\n" + "=" * 60)
    print("测试 3: Checkpoint加载验证")
    print("=" * 60)
    
    device = torch.device('cpu')
    BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(BACKEND_ROOT, 'checkpoints')
    
    # 检查文件是否存在
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    tst1_path = os.path.join(checkpoint_dir, 'tst1_best.pt')
    tst2_path = os.path.join(checkpoint_dir, 'tst2_best.pt')
    
    print(f"检查checkpoint文件:")
    print(f"  - best_model.pt: {'存在' if os.path.exists(best_model_path) else '不存在'}")
    print(f"  - tst1_best.pt: {'存在' if os.path.exists(tst1_path) else '不存在'}")
    print(f"  - tst2_best.pt: {'存在' if os.path.exists(tst2_path) else '不存在'}")
    
    # 测试1: 从best_model.pt加载（包含键名转换）
    if os.path.exists(best_model_path):
        print(f"\n测试从 best_model.pt 加载:")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 转换键名
        new_state_dict = {}
        tst1_keys = 0
        tst2_keys = 0
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('tst1.'):
                new_key = key.replace('tst1.', 'transformer_ts.', 1)
                tst1_keys += 1
            elif key.startswith('tst2.'):
                new_key = key.replace('tst2.', 'transformer_fc.', 1)
                tst2_keys += 1
            new_state_dict[new_key] = value
        
        print(f"  - 原始键总数: {len(state_dict)}")
        print(f"  - TST1键数量: {tst1_keys}")
        print(f"  - TST2键数量: {tst2_keys}")
        
        # 过滤尺寸不匹配的键
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for key, value in new_state_dict.items():
            if key in model_state_dict and value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
        
        print(f"  - 转换后键总数: {len(new_state_dict)}")
        print(f"  - 尺寸匹配的键: {len(filtered_state_dict)}")
        
        # 检查关键参数
        critical_keys = [
            'transformer_fc.input_embedding.weight',
            'transformer_ts.input_embedding.weight'
        ]
        
        for key in critical_keys:
            if key in filtered_state_dict:
                print(f"  [PASS] {key} 已加载")
            else:
                print(f"  [FAIL] {key} 未找到或尺寸不匹配")
        
        log_debug("test_model_alignment.py:test_checkpoint_loading", "best_model.pt loading", {
            "total_keys": len(state_dict),
            "tst1_keys": tst1_keys,
            "tst2_keys": tst2_keys,
            "filtered_keys": len(filtered_state_dict),
            "transformer_fc_input_embedding_loaded": 'transformer_fc.input_embedding.weight' in filtered_state_dict,
            "transformer_ts_input_embedding_loaded": 'transformer_ts.input_embedding.weight' in filtered_state_dict
        })
    
    # 测试2: 从单独的tst1_best.pt和tst2_best.pt加载
    if os.path.exists(tst1_path) and os.path.exists(tst2_path):
        print(f"\n测试从单独checkpoint文件加载:")
        
        # 测试TST1加载
        print(f"  测试TST1加载...")
        try:
            tst1_checkpoint = torch.load(tst1_path, map_location=device, weights_only=False)
            tst1_state_dict = tst1_checkpoint.get('model_state_dict', tst1_checkpoint.get('state_dict', tst1_checkpoint))
            print(f"    - TST1 checkpoint键数量: {len(tst1_state_dict)}")
            
            # 尝试加载到transformer_ts
            tst1_keys_in_model = [k for k in tst1_state_dict.keys() if k in model.transformer_ts.state_dict()]
            print(f"    - 与模型匹配的键: {len(tst1_keys_in_model)}")
            
            if len(tst1_keys_in_model) > 0:
                print(f"    [PASS] TST1 checkpoint与模型兼容")
            else:
                print(f"    [FAIL] TST1 checkpoint键名不匹配")
        except Exception as e:
            print(f"    [FAIL] TST1加载失败: {e}")
        
        # 测试TST2加载
        print(f"  测试TST2加载...")
        try:
            tst2_checkpoint = torch.load(tst2_path, map_location=device, weights_only=False)
            tst2_state_dict = tst2_checkpoint.get('model_state_dict', tst2_checkpoint.get('state_dict', tst2_checkpoint))
            print(f"    - TST2 checkpoint键数量: {len(tst2_state_dict)}")
            
            # 检查input_embedding维度
            tst2_input_emb_key = None
            for key in tst2_state_dict.keys():
                if 'input_embedding' in key and 'weight' in key:
                    tst2_input_emb_key = key
                    break
            
            if tst2_input_emb_key:
                tst2_input_emb_shape = tst2_state_dict[tst2_input_emb_key].shape
                print(f"    - TST2 input_embedding形状: {tst2_input_emb_shape}")
                
                model_input_emb_shape = model.transformer_fc.input_embedding.weight.shape
                if tst2_input_emb_shape == model_input_emb_shape:
                    print(f"    [PASS] TST2 input_embedding维度匹配: {tst2_input_emb_shape}")
                else:
                    print(f"    [FAIL] TST2 input_embedding维度不匹配！checkpoint {tst2_input_emb_shape} vs 模型 {model_input_emb_shape}")
            
            # 尝试加载到transformer_fc
            tst2_keys_in_model = [k for k in tst2_state_dict.keys() if k in model.transformer_fc.state_dict()]
            print(f"    - 与模型匹配的键: {len(tst2_keys_in_model)}")
            
            if len(tst2_keys_in_model) > 0:
                print(f"    [PASS] TST2 checkpoint与模型兼容")
            else:
                print(f"    [FAIL] TST2 checkpoint键名不匹配")
                
            log_debug("test_model_alignment.py:test_checkpoint_loading", "separate checkpoints loading", {
                "tst1_keys": len(tst1_state_dict),
                "tst2_keys": len(tst2_state_dict),
                "tst2_input_embedding_shape": list(tst2_input_emb_shape) if tst2_input_emb_key else None,
                "model_input_embedding_shape": list(model_input_emb_shape)
            })
        except Exception as e:
            print(f"    [FAIL] TST2加载失败: {e}")
    
    return True


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 4: 前向传播验证")
    print("=" * 60)
    
    # 模拟数据
    batch_size = 2
    window_size = 32
    n_rois = 200
    pcc_dim = 200 * 199 // 2  # 19900
    
    timeseries = torch.randn(batch_size, window_size, n_rois)
    pcc_vectors = torch.randn(batch_size, pcc_dim)
    
    print(f"输入数据:")
    print(f"  - 时间序列: {timeseries.shape} (batch, 窗口大小, 脑区)")
    print(f"  - PCC向量: {pcc_vectors.shape} (batch, PCC维度)")
    
    model.eval()
    with torch.no_grad():
        try:
            logits = model(timeseries, pcc_vectors)
            print(f"\n  [PASS] 前向传播成功！")
            print(f"  - 输出logits形状: {logits.shape} (batch, num_classes)")
            
            log_debug("test_model_alignment.py:test_forward_pass", "Forward pass success", {
                "input_timeseries_shape": list(timeseries.shape),
                "input_pcc_shape": list(pcc_vectors.shape),
                "output_logits_shape": list(logits.shape)
            })
            return True
        except Exception as e:
            print(f"\n  [FAIL] 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            log_debug("test_model_alignment.py:test_forward_pass", "Forward pass failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("模型对齐测试脚本")
    print("=" * 60)
    
    results = []
    
    # 测试1: 数据流
    if test_data_flow():
        results.append(("数据流", True))
    else:
        results.append(("数据流", False))
        return
    
    # 测试2: 模型结构
    success, model = test_model_structure()
    if success:
        results.append(("模型结构", True))
    else:
        results.append(("模型结构", False))
        return
    
    # 测试3: Checkpoint加载
    if test_checkpoint_loading(model):
        results.append(("Checkpoint加载", True))
    else:
        results.append(("Checkpoint加载", False))
    
    # 测试4: 前向传播
    if test_forward_pass(model):
        results.append(("前向传播", True))
    else:
        results.append(("前向传播", False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, success in results:
        status = "[PASS] 通过" if success else "[FAIL] 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n[PASS] 所有测试通过！")
    else:
        print("\n[FAIL] 部分测试失败，请检查上述输出")


if __name__ == '__main__':
    main()
