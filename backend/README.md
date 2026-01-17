# TwoTST API Server

TwoTST的Flask后端服务器，提供fMRI数据的预测和异常连接分析服务。

## 目录

- [功能概述](#功能概述)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [使用示例](#使用示例)
- [技术架构](#技术架构)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

## 功能概述

TwoTST API Server 是一个基于Flask的RESTful API服务，提供以下功能：

1. **fMRI数据预测** (`/predict`): 
   - 接收用户的fMRI ROI数据（CSV/JSON格式）
   - 自动对齐到CC200标准（200个ROI）
   - 使用滑动窗口切分数据（window_size=32, stride=16）
   - 加载训练好的模型进行预测
   - 返回每个窗口的ASD概率

2. **异常连接分析** (`/analyze`):
   - 在预测基础上，分析CC200脑区的异常连接模式
   - 对比ASD窗口和TC窗口的PCC连接强度差异
   - 识别top-k差异最大的ROI连接对
   - 返回详细的异常连接统计信息

3. **健康检查** (`/health`): 服务状态检查

4. **CC200 ROI信息** (`/cc200/rois`): 获取CC200的ROI标签列表

## 快速开始

### 1. 环境要求

- Python 3.7+
- PyTorch 1.9+
- Flask 2.0+
- CUDA (可选，用于GPU加速)

### 2. 安装依赖

```bash
cd /root/workplace/exp/TwoTST/api
pip install -r requirements.txt
```

### 3. 启动服务器

```bash
# 默认配置（localhost:5000）
python app.py

# 自定义配置
python app.py --host 0.0.0.0 --port 5000 --debug

# 后台运行（推荐生产环境）
nohup python app.py --host 0.0.0.0 --port 5000 > server.log 2>&1 &
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:5000/health

# 预期返回
{
  "status": "healthy",
  "service": "TwoTST API Server",
  "version": "1.0.0"
}
```

## API文档

### 1. 健康检查

**端点**: `GET /health`

**描述**: 检查服务运行状态

**请求示例**:
```bash
curl http://localhost:5000/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "service": "TwoTST API Server",
  "version": "1.0.0"
}
```

---

### 2. 预测接口

**端点**: `POST /predict`

**描述**: 接收fMRI数据，返回每个窗口的ASD预测概率

**请求方式**:
- **方式1**: JSON Body
  ```bash
  Content-Type: application/json
  Body: {"timeseries": [[...], [...]]} 或 {"data": [[...], [...]]}
  ```
- **方式2**: 文件上传
  ```bash
  Content-Type: multipart/form-data
  Form: file=@data.csv 或 file=@data.json
  ```

**查询参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `window_size` | int | 32 | 滑动窗口大小 |
| `stride` | int | 16 | 滑动窗口步长 |
| `batch_size` | int | 32 | 预测批次大小 |
| `align_cc200` | bool | true | 是否对齐到CC200（200个ROI） |
| `return_attention` | bool | false | 是否返回注意力权重 |
| `checkpoint` | str | (默认路径) | 模型checkpoint路径（可选） |
| `results_json` | str | (默认路径) | results.json路径（可选） |
| `device` | str | cuda | 计算设备（cuda/cpu） |

**请求示例**:

```bash
# JSON请求
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timeseries": [
      [0.1, 0.2, 0.3, ...],  # 时间点1的200个ROI
      [0.2, 0.3, 0.4, ...],  # 时间点2的200个ROI
      ...
    ]
  }' \
  -G --data-urlencode "window_size=32" \
  --data-urlencode "stride=16"
```

```bash
# 文件上传
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/data.csv" \
  -G --data-urlencode "window_size=32"
```

**响应格式**:
```json
{
  "success": true,
  "predictions": {
    "window_predictions": [
      {
        "window_index": 0,
        "asd_probability": 0.8523,
        "prediction": 1,
        "logits": [0.5234, 1.2345]
      },
      {
        "window_index": 1,
        "asd_probability": 0.7456,
        "prediction": 1,
        "logits": [0.4123, 1.1234]
      },
      ...
    ],
    "summary": {
      "total_windows": 10,
      "mean_asd_probability": 0.7890,
      "std_asd_probability": 0.1234,
      "predicted_asd_windows": 8,
      "predicted_tc_windows": 2
    }
  },
  "data_info": {
    "n_rois": 200,
    "n_timepoints": 100,
    "n_windows": 10
  }
}
```

**字段说明**:
- `window_predictions`: 每个窗口的预测结果列表
  - `window_index`: 窗口索引
  - `asd_probability`: ASD概率（0-1之间）
  - `prediction`: 预测类别（0=TC, 1=ASD）
  - `logits`: 模型原始输出
- `summary`: 预测汇总统计
- `data_info`: 数据信息

---

### 3. 异常连接分析接口

**端点**: `POST /analyze`

**描述**: 接收fMRI数据，返回预测结果和CC200脑区的异常连接分析

**请求方式**: 同 `/predict`

**查询参数**: 同 `/predict`，额外参数：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.5 | ASD概率阈值（用于区分ASD和TC窗口） |
| `top_k` | int | 50 | 返回top-k异常连接 |

**请求示例**:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"timeseries": [[...], [...]]}' \
  -G --data-urlencode "window_size=32" \
  --data-urlencode "top_k=50" \
  --data-urlencode "threshold=0.5"
```

**响应格式**:
```json
{
  "success": true,
  "predictions": {
    ...  // 同 /predict 响应
  },
  "abnormal_connections": {
    "abnormal_connections": [
      {
        "roi_i": 10,
        "roi_j": 45,
        "difference": 0.1523,
        "asd_connection_strength": 0.8234,
        "tc_connection_strength": 0.6711
      },
      {
        "roi_i": 23,
        "roi_j": 67,
        "difference": 0.1456,
        "asd_connection_strength": 0.7890,
        "tc_connection_strength": 0.6434
      },
      ...
    ],
    "connection_statistics": {
      "asd": {
        "mean_connection_strength": 0.7523,
        "std_connection_strength": 0.1234,
        "n_windows": 8
      },
      "tc": {
        "mean_connection_strength": 0.6456,
        "std_connection_strength": 0.1123,
        "n_windows": 2
      }
    },
    "summary": {
      "n_rois": 200,
      "n_abnormal_connections_identified": 50,
      "threshold_used": 0.5,
      "asd_windows_count": 8,
      "tc_windows_count": 2
    }
  },
  "data_info": {
    "n_rois": 200,
    "n_timepoints": 100,
    "n_windows": 10
  }
}
```

**字段说明**:
- `abnormal_connections`: 异常连接列表（按差异从大到小排序）
  - `roi_i`, `roi_j`: ROI索引（CC200对齐后的索引，0-199）
  - `difference`: 连接强度差异（绝对值）
  - `asd_connection_strength`: ASD组的平均连接强度
  - `tc_connection_strength`: TC组的平均连接强度
- `connection_statistics`: 连接统计信息
- `summary`: 分析汇总

---

### 4. 获取CC200 ROI信息

**端点**: `GET /cc200/rois`

**描述**: 获取CC200的ROI标签列表

**请求示例**:
```bash
curl http://localhost:5000/cc200/rois
```

**响应格式**:
```json
{
  "success": true,
  "rois": [
    {"roi_index": 0, "roi_name": "ROI_000"},
    {"roi_index": 1, "roi_name": "ROI_001"},
    ...
  ],
  "total_rois": 200
}
```

## 使用示例

### Python示例

```python
import requests
import json
import numpy as np

# 服务器地址
BASE_URL = "http://localhost:5000"

# 1. 准备测试数据（100个时间点，200个ROI）
timeseries = np.random.randn(100, 200).tolist()

# 2. 健康检查
response = requests.get(f"{BASE_URL}/health")
print("Health check:", response.json())

# 3. 预测
response = requests.post(
    f"{BASE_URL}/predict",
    json={"timeseries": timeseries},
    params={
        "window_size": 32,
        "stride": 16,
        "batch_size": 32
    }
)
result = response.json()
print("\n=== 预测结果 ===")
print(f"总窗口数: {result['predictions']['summary']['total_windows']}")
print(f"平均ASD概率: {result['predictions']['summary']['mean_asd_probability']:.4f}")
for pred in result['predictions']['window_predictions'][:5]:
    print(f"窗口 {pred['window_index']}: ASD概率={pred['asd_probability']:.4f}")

# 4. 异常连接分析
response = requests.post(
    f"{BASE_URL}/analyze",
    json={"timeseries": timeseries},
    params={
        "window_size": 32,
        "stride": 16,
        "top_k": 20,
        "threshold": 0.5
    }
)
result = response.json()
print("\n=== 异常连接分析 ===")
abnormal = result['abnormal_connections']['abnormal_connections']
print(f"识别到 {len(abnormal)} 个异常连接")
print("\nTop-5异常连接:")
for conn in abnormal[:5]:
    print(f"  ROI {conn['roi_i']} <-> ROI {conn['roi_j']}: "
          f"差异={conn['difference']:.4f} "
          f"(ASD={conn['asd_connection_strength']:.4f}, "
          f"TC={conn['tc_connection_strength']:.4f})")
```

### cURL示例

```bash
# 1. 健康检查
curl http://localhost:5000/health

# 2. JSON预测
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timeseries": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
  }' \
  -G --data-urlencode "window_size=32"

# 3. CSV文件上传
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/data.csv" \
  -G --data-urlencode "window_size=32"

# 4. 异常连接分析
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"timeseries": [[...], [...]]}' \
  -G --data-urlencode "top_k=50"
```

## 技术架构

### 目录结构

```
api/
├── __init__.py
├── app.py                          # Flask主应用（入口文件）
├── requirements.txt                # 依赖包列表
├── README.md                       # 本文档
├── utils/
│   └── data_processor.py          # 数据处理器
│       - 数据加载（JSON/CSV/ndarray）
│       - CC200对齐
│       - 滑动窗口切分
│       - PCC计算
├── services/
│   ├── prediction_service.py      # 预测服务
│   │   - 模型加载
│   │   - 批量预测
│   │   - 概率计算
│   └── connection_analysis_service.py  # 异常连接分析服务
│       - PCC向量到连接矩阵转换
│       - ASD/TC窗口分离
│       - 异常连接识别
└── uploads/                        # 临时上传文件目录
```

### 数据流程

```
用户输入 (JSON/CSV)
  ↓
数据处理器 (data_processor.py)
  ├─ 加载数据
  ├─ CC200对齐 (200个ROI)
  ├─ 滑动窗口切分 (window_size=32, stride=16)
  └─ PCC计算 (19900维向量)
  ↓
预测服务 (prediction_service.py)
  ├─ 加载模型checkpoint
  ├─ 批量预测
  └─ 计算ASD概率
  ↓
异常连接分析服务 (connection_analysis_service.py)
  ├─ PCC向量 → 连接矩阵 (200x200)
  ├─ ASD/TC窗口分离
  ├─ 连接差异计算
  └─ Top-k异常连接识别
  ↓
返回结果 (JSON)
```

### 核心组件

1. **数据处理器** (`utils/data_processor.py`)
   - `process_input_data()`: 主处理函数
   - `load_fmri_data_from_json()`: JSON数据加载
   - `load_fmri_data_from_csv()`: CSV数据加载
   - `align_to_cc200()`: CC200对齐
   - `apply_sliding_window()`: 滑动窗口切分
   - `compute_pcc()`: PCC计算

2. **预测服务** (`services/prediction_service.py`)
   - `PredictionService`: 预测服务类
   - `_load_model()`: 模型加载
   - `predict()`: 批量预测

3. **异常连接分析服务** (`services/connection_analysis_service.py`)
   - `ConnectionAnalysisService`: 分析服务类
   - `analyze_abnormal_connections()`: 异常连接分析
   - `pcc_vector_to_matrix()`: PCC向量转连接矩阵

## 配置说明

### 模型配置

默认使用以下模型：
- **Checkpoint**: `checkpoints_sw/finetune/sw_baseline_cross_attention/best_model.pt`
- **配置**: `checkpoints_sw/finetune/sw_baseline_cross_attention/results.json`

可通过查询参数自定义：
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timeseries": [[...]]}' \
  -G --data-urlencode "checkpoint=/path/to/model.pt" \
  --data-urlencode "results_json=/path/to/results.json"
```

### 服务器配置

启动参数：
```bash
python app.py --host 0.0.0.0 --port 5000 [--debug]
```

环境变量（可选）：
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

### 数据格式要求

**输入数据格式**:
- **JSON**: 
  ```json
  {"timeseries": [[...], [...]]}  // (n_timepoints, n_rois)
  或
  {"data": [[...], [...]]}
  ```
- **CSV**: 
  - 每行是一个时间点
  - 每列是一个ROI
  - 第一行为列名（可选）

**ROI数量**:
- 输入ROI数量可以是任意值（推荐100-300）
- 系统会自动对齐到CC200（200个ROI）
  - ROI数 < 200: 零填充
  - ROI数 = 200: 直接使用
  - ROI数 > 200: 均匀采样

**时间点数量**:
- 建议 ≥ 100个时间点
- 窗口大小为32，步长为16时，需要 ≥ 32个时间点

## 常见问题

### Q1: 如何对齐到CC200？

A: 系统会自动对齐。输入数据的ROI数量可以是任意值，系统会：
- ROI数 < 200: 零填充到200
- ROI数 = 200: 直接使用
- ROI数 > 200: 均匀采样到200

### Q2: 滑动窗口参数如何设置？

A: 建议与训练时一致：
- `window_size`: 32（训练时使用的窗口大小）
- `stride`: 16（训练时使用的步长）

### Q3: 如何理解异常连接的ROI索引？

A: ROI索引是CC200对齐后的索引（0-199）。`roi_i`和`roi_j`表示连接的两个ROI。

### Q4: 如何处理多个受试者的数据？

A: 目前API每次只处理一个受试者。多个受试者需要分别调用API。

### Q5: 模型加载失败怎么办？

A: 检查：
1. Checkpoint文件路径是否正确
2. results.json路径是否正确
3. 模型文件是否完整
4. CUDA是否可用（如果使用GPU）

### Q6: 内存不足怎么办？

A: 
1. 减小`batch_size`参数
2. 使用CPU模式（`device=cpu`）
3. 减少时间点数量

### Q7: 如何查看详细的错误信息？

A: 启动时使用`--debug`参数：
```bash
python app.py --debug
```

## 许可证

与TwoTST项目保持一致。

## 联系与支持

如有问题或建议，请参考TwoTST项目主README或提交Issue。