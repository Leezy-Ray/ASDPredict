"""测试classifier配置"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.prediction_service import PredictionService

checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
results_json_path = os.path.join('checkpoints', 'results.json')

ps = PredictionService(checkpoint_path, results_json_path, 'cpu')

print("Classifier结构:")
for i, layer in enumerate(ps.model.classifier):
    print(f"  {i}: {layer}")

print("\nClassifier参数:")
for name, param in ps.model.classifier.named_parameters():
    print(f"  {name}: {param.shape}")

# 检查checkpoint中的classifier键
import torch
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

classifier_keys = [k for k in state_dict.keys() if 'classifier' in k.lower()]
print(f"\nCheckpoint中的classifier键 ({len(classifier_keys)}):")
for key in classifier_keys:
    print(f"  {key}: {state_dict[key].shape}")
