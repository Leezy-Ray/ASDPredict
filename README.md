# ASD 脑连接预测系统

基于 fMRI 数据的自闭症谱系障碍（ASD）风险评估与脑区异常连接可视化系统。
<img width="1854" height="854" alt="image" src="https://github.com/user-attachments/assets/d70653f3-2e5e-479c-926b-bb069b404f96" />


## 项目结构

```
ASDModelPredict/
├── data/              # 数据目录
│   ├── checkpoints/   # 模型检查点文件
│   ├── cc200_coordinates.json  # CC200 脑区坐标数据
│   ├── processed_data.pkl      # 处理后的数据文件
│   ├── craddock_2011_parcellations.tar.gz  # Craddock 分区数据
│   └── craddock_extracted/     # 解压后的分区数据
├── backend/           # 后端服务（Flask API）
│   ├── app.py         # Flask 应用入口
│   ├── models/        # 深度学习模型
│   ├── services/      # 业务逻辑服务
│   ├── utils/         # 工具函数
│   ├── scripts/       # 数据处理脚本
│   ├── requirements.txt
│   └── README.md
└── web/               # 前端应用（Next.js）
    ├── src/           # 源代码
    ├── public/        # 静态资源
    ├── package.json
    └── README.md
```

## 功能特性

- 📊 **fMRI 数据预测** - 基于 TwoTST 模型的 ASD 风险评估
- 🧠 **3D 大脑可视化** - 交互式脑区异常连接可视化
- 🔗 **异常连接分析** - CC200 标准脑区的连接模式分析
- 📈 **实时预测** - 滑动窗口预测与结果展示

## 快速开始

### 环境要求

- Python 3.7+ (后端)
- Node.js 18+ 或 Bun 1.0+ (前端)
- PyTorch 1.9+ (模型推理)

### 后端启动

```bash
cd backend
pip install -r requirements.txt
python app.py
```

详细文档请参考 [backend/README.md](backend/README.md)

### 前端启动

```bash
cd web
bun install
bun dev
```

详细文档请参考 [web/README.md](web/README.md)

## 数据目录说明

所有数据文件统一存放在 `data/` 目录中：

- **checkpoints/**: 训练好的模型检查点文件（不包含在 Git 中）
- **cc200_coordinates.json**: CC200 标准脑区坐标数据
- **processed_data.pkl**: 预处理后的 fMRI 数据
- **craddock_2011_parcellations.tar.gz**: Craddock 2011 分区数据压缩包
- **craddock_extracted/**: 解压后的 Craddock 分区数据

## 技术栈

### 后端
- Flask - Web 框架
- PyTorch - 深度学习框架
- NumPy, SciPy - 数据处理

### 前端
- Next.js 16 - React 框架
- Three.js - 3D 可视化
- TypeScript - 类型安全
- Tailwind CSS - 样式框架
- Zustand - 状态管理

## 开发说明

### 数据文件管理

- 所有数据文件应存放在 `data/` 目录
- 大文件（>100MB）不会被 Git 跟踪，请使用 Git LFS 或单独存储
- 模型检查点文件存放在 `data/checkpoints/`

### 代码规范

- 后端使用 Python，遵循 PEP 8 规范
- 前端使用 TypeScript，遵循 ESLint 规则
- 单文件代码超过 500 行应考虑拆分

## 许可证

本项目为私有项目。

## 相关链接

- [后端 API 文档](backend/README.md)
- [前端开发文档](web/README.md)
