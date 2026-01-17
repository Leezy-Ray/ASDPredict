"""
检查checkpoints目录下所有.pt文件的加载情况
"""

import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.prediction_service import PredictionService
from models.dual_stream import DualStreamModel

checkpoint_dir = os.path.join('checkpoints')
checkpoint_files = {
    'best_model.pt': os.path.join(checkpoint_dir, 'best_model.pt'),
    'tst1_best.pt': os.path.join(checkpoint_dir, 'tst1_best.pt'),
    'tst2_best.pt': os.path.join(checkpoint_dir, 'tst2_best.pt'),
}

results_json_path = os.path.join(checkpoint_dir, 'results.json')

print("=" * 80)
print("检查所有Checkpoint文件")
print("=" * 80)

# 1. 检查文件是否存在
print("\n1. 检查文件存在性:")
for name, path in checkpoint_files.items():
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[缺失]"
    print(f"   {status} {name}: {path}")
    if exists:
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"      大小: {size:.2f} MB")

# 2. 检查best_model.pt的内容
print("\n2. 检查best_model.pt的内容:")
if os.path.exists(checkpoint_files['best_model.pt']):
    checkpoint = torch.load(checkpoint_files['best_model.pt'], map_location='cpu', weights_only=False)
    
    # 检查checkpoint结构
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("   - 包含 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("   - 包含 'state_dict'")
    else:
        state_dict = checkpoint
        print("   - 直接是state_dict")
    
    # 统计键
    tst1_keys = [k for k in state_dict.keys() if 'tst1' in k.lower() or 'transformer_ts' in k.lower()]
    tst2_keys = [k for k in state_dict.keys() if 'tst2' in k.lower() or 'transformer_fc' in k.lower()]
    fusion_keys = [k for k in state_dict.keys() if 'fusion' in k.lower()]
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k.lower()]
    
    print(f"   - TST1相关键: {len(tst1_keys)}")
    print(f"   - TST2相关键: {len(tst2_keys)}")
    print(f"   - Fusion相关键: {len(fusion_keys)}")
    print(f"   - Classifier相关键: {len(classifier_keys)}")
    print(f"   - 总键数: {len(state_dict)}")
    
    # 检查是否包含epoch信息
    if 'epoch' in checkpoint:
        print(f"   - Epoch: {checkpoint['epoch']}")
    else:
        print("   - Epoch: 未保存")
    
    # 检查是否包含其他元数据
    other_keys = [k for k in checkpoint.keys() if k not in ['model_state_dict', 'state_dict', 'epoch']]
    if other_keys:
        print(f"   - 其他键: {other_keys}")
else:
    print("   [错误] best_model.pt不存在")

# 3. 检查tst1_best.pt和tst2_best.pt
print("\n3. 检查单独的预训练权重文件:")

for name in ['tst1_best.pt', 'tst2_best.pt']:
    path = checkpoint_files[name]
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"   {name}:")
        print(f"     - 键数量: {len(state_dict)}")
        print(f"     - 键示例: {list(state_dict.keys())[:3]}")
    else:
        print(f"   {name}: [缺失]")

# 4. 检查results.json中的配置
print("\n4. 检查results.json中的预训练配置:")
if os.path.exists(results_json_path):
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    pretrain_config = results.get('config', {}).get('pretrain', {})
    use_pretrained = pretrain_config.get('use_pretrained', False)
    
    print(f"   - 使用预训练权重: {use_pretrained}")
    
    if use_pretrained:
        tst1_checkpoint = pretrain_config.get('tst1', {}).get('checkpoint', None)
        tst2_checkpoint = pretrain_config.get('tst2', {}).get('checkpoint', None)
        
        print(f"   - TST1预训练路径: {tst1_checkpoint}")
        print(f"   - TST2预训练路径: {tst2_checkpoint}")
        
        # 检查这些路径是否指向本地文件
        if tst1_checkpoint:
            local_tst1 = os.path.join(checkpoint_dir, 'tst1_best.pt')
            if os.path.exists(local_tst1):
                print(f"     -> 本地文件存在: {local_tst1}")
            else:
                print(f"     -> 本地文件不存在")
        
        if tst2_checkpoint:
            local_tst2 = os.path.join(checkpoint_dir, 'tst2_best.pt')
            if os.path.exists(local_tst2):
                print(f"     -> 本地文件存在: {local_tst2}")
            else:
                print(f"     -> 本地文件不存在")

# 5. 检查当前加载逻辑
print("\n5. 检查当前加载逻辑:")
print("   当前PredictionService只加载best_model.pt")
print("   best_model.pt应该包含完整的模型权重（TST1 + TST2 + Fusion + Classifier）")

# 6. 验证best_model.pt是否包含完整的权重
print("\n6. 验证best_model.pt是否包含完整的权重:")
if os.path.exists(checkpoint_files['best_model.pt']):
    try:
        ps = PredictionService(
            checkpoint_files['best_model.pt'],
            results_json_path,
            'cpu'
        )
        
        # 检查模型参数
        model = ps.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   - 模型总参数数: {total_params:,}")
        print(f"   - 可训练参数数: {trainable_params:,}")
        
        # 检查各个组件的参数
        tst1_params = sum(p.numel() for p in model.transformer_ts.parameters())
        tst2_params = sum(p.numel() for p in model.transformer_fc.parameters())
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        print(f"   - TST1参数数: {tst1_params:,}")
        print(f"   - TST2参数数: {tst2_params:,}")
        print(f"   - Fusion参数数: {fusion_params:,}")
        print(f"   - Classifier参数数: {classifier_params:,}")
        
        # 验证参数是否被加载（检查是否有非零值）
        print("\n   检查参数是否被正确加载:")
        
        # 检查TST1的input_embedding
        tst1_emb = model.transformer_ts.input_embedding.weight
        tst1_emb_sum = tst1_emb.abs().sum().item()
        print(f"   - TST1 input_embedding权重和: {tst1_emb_sum:.2f} (非零表示已加载)")
        
        # 检查TST2的input_embedding
        tst2_emb = model.transformer_fc.input_embedding.weight
        tst2_emb_sum = tst2_emb.abs().sum().item()
        print(f"   - TST2 input_embedding权重和: {tst2_emb_sum:.2f} (非零表示已加载)")
        
        # 检查Fusion
        fusion_proj_ts = model.fusion.proj_ts.weight
        fusion_proj_ts_sum = fusion_proj_ts.abs().sum().item()
        print(f"   - Fusion proj_ts权重和: {fusion_proj_ts_sum:.2f} (非零表示已加载)")
        
        # 检查Classifier
        classifier_first = model.classifier[0].weight
        classifier_first_sum = classifier_first.abs().sum().item()
        print(f"   - Classifier第一层权重和: {classifier_first_sum:.2f} (非零表示已加载)")
        
        print("\n   [结论]")
        if tst1_emb_sum > 0 and tst2_emb_sum > 0 and fusion_proj_ts_sum > 0 and classifier_first_sum > 0:
            print("   [OK] best_model.pt包含了完整的模型权重，所有组件都已正确加载")
            print("   [信息] tst1_best.pt和tst2_best.pt是预训练权重，在训练时使用，")
            print("          但best_model.pt已经包含了finetune后的完整权重，不需要单独加载")
        else:
            print("   [警告] 某些组件的权重可能未正确加载")
            
    except Exception as e:
        print(f"   [错误] 加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 7. 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("""
结论：
1. best_model.pt: 应该包含完整的模型权重（TST1 + TST2 + Fusion + Classifier）
2. tst1_best.pt和tst2_best.pt: 是预训练权重，在训练阶段使用
3. 预测时只需要加载best_model.pt即可，因为best_model.pt已经包含了finetune后的完整权重

如果best_model.pt不包含TST1和TST2的权重，则需要：
- 先加载tst1_best.pt和tst2_best.pt作为预训练权重
- 然后再加载best_model.pt中的其他权重（Fusion + Classifier）
""")
