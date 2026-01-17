# ASD 脑连接预测系统 - Web 前端

基于 Next.js 的 ASD（自闭症谱系障碍）风险评估与脑区异常连接可视化前端应用。

## 功能特性

- 📊 **fMRI 数据上传与分析** - 支持 CSV/JSON 格式的 fMRI ROI 数据上传
- 🧠 **3D 大脑可视化** - 使用 Three.js 实现交互式 3D 大脑模型展示
- 🔗 **异常连接分析** - 可视化 CC200 标准脑区的异常连接模式
- 📈 **预测结果展示** - 实时显示 ASD 风险评估结果和窗口预测数据
- 🎯 **区域信息面板** - 显示选定脑区的详细信息和连接强度

## 技术栈

- **框架**: Next.js 16.1.2 (App Router)
- **运行时**: React 19.2.3
- **语言**: TypeScript 5
- **样式**: Tailwind CSS 4
- **3D 渲染**: Three.js + React Three Fiber + Drei
- **状态管理**: Zustand
- **数据处理**: PapaParse
- **包管理器**: Bun

## 快速开始

### 环境要求

- Node.js 18+ 或 Bun 1.0+
- 推荐使用 Bun 作为包管理器

### 安装依赖

```bash
# 使用 bun 安装依赖
bun install
```

### 开发模式

```bash
# 启动开发服务器（默认端口 3000）
bun dev
```

打开浏览器访问 [http://localhost:3000](http://localhost:3000)

### 构建生产版本

```bash
# 构建生产版本
bun build

# 启动生产服务器
bun start
```

### 代码检查

```bash
# 运行 ESLint
bun lint
```

## 项目结构

```
web/
├── public/                 # 静态资源
│   └── cc200_coordinates.json  # CC200 脑区坐标数据
├── src/
│   ├── app/               # Next.js App Router
│   │   ├── api/           # API 路由
│   │   │   └── predict/   # 预测 API
│   │   ├── layout.tsx     # 根布局
│   │   └── page.tsx       # 首页
│   ├── components/        # React 组件
│   │   ├── BrainViewer.tsx      # 3D 大脑可视化组件
│   │   ├── BrainModel.tsx       # 大脑模型组件
│   │   ├── FileUploader.tsx     # 文件上传组件
│   │   ├── ResultDisplay.tsx    # 结果展示组件
│   │   ├── ControlPanel.tsx     # 控制面板组件
│   │   └── ...
│   ├── lib/               # 工具库
│   │   ├── cc200-regions.ts     # CC200 区域定义
│   │   └── useCC200Regions.ts   # CC200 区域 Hook
│   └── store/             # 状态管理
│       └── useStore.ts    # Zustand store
├── next.config.ts         # Next.js 配置
├── tsconfig.json          # TypeScript 配置
├── package.json           # 项目依赖
└── README.md              # 项目文档
```

## 核心组件说明

### BrainViewer
3D 大脑可视化主组件，使用 React Three Fiber 渲染交互式大脑模型。

### FileUploader
支持 CSV/JSON 格式的 fMRI 数据文件上传组件。

### ResultDisplay
展示 ASD 预测结果和窗口预测数据。

### ControlPanel
提供可视化控制选项，如颜色映射、连接阈值等。

## 环境变量

在项目根目录创建 `.env.local` 文件配置环境变量：

```env
# 后端 API 地址
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## API 集成

前端通过 `/api/predict` 路由与后端 Flask API 服务通信，进行：
- fMRI 数据预测
- 异常连接分析
- CC200 ROI 信息获取

详细的后端 API 文档请参考 `../backend/README.md`。

## 开发注意事项

1. **Three.js 配置**: 项目在 `next.config.ts` 中配置了 `transpilePackages: ['three']` 以支持 Three.js 在 Next.js 中使用。

2. **动态导入**: BrainViewer 组件使用动态导入以避免 SSR 问题，因为 Three.js 依赖浏览器环境。

3. **文件大小**: 如果单个文件超过 500 行代码，建议拆分成多个文件（参考用户规则）。

4. **不使用 setTimeout**: 除非用户明确授权，否则不要使用 `setTimeout`。

## 浏览器兼容性

- Chrome/Edge (最新版本)
- Firefox (最新版本)
- Safari (最新版本)

## 许可证

本项目为私有项目。

## 相关链接

- [后端 API 文档](../backend/README.md)
- [Next.js 文档](https://nextjs.org/docs)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)
- [Bun 文档](https://bun.sh/docs)
