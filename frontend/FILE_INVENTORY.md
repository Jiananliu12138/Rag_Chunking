# RAG Platform 前端文件清单

## 📦 项目统计

- **总页面数**: 6 个
- **核心组件**: 2 个（Layout + ImageWithFallback）
- **UI 组件**: 16 个（实际使用）
- **工具函数**: 1 个（api.ts）
- **样式文件**: 2 个

---

## 📁 完整文件列表

### `/` 根目录
```
├── index.html                    # HTML 入口
├── package.json                  # 完整依赖（含未使用的）
├── MINIMAL_PACKAGE.json          # 最小化依赖（仅核心）
├── CLEANUP_GUIDE.md              # 清理指南
├── FILE_INVENTORY.md             # 本文件
├── migrate.sh                    # 迁移脚本
├── vite.config.ts                # Vite 配置
└── tsconfig.json                 # TypeScript 配置
```

### `/src/` 源代码目录

#### `/src/app/` - 主应用
```
src/app/
├── App.tsx                       # 应用入口，RouterProvider
├── routes.ts                     # 路由配置
├── pages/                        # 6 个页面
│   ├── Home.tsx                  # ⭐ RAG 聊天主页（3.2 KB）
│   ├── ChunkingPage.tsx          # 分块页面（10.5 KB）
│   ├── IndexPage.tsx             # 索引管理页面（15.8 KB）
│   ├── RetrievalPage.tsx         # 检索页面（12.3 KB）
│   ├── EvalPage.tsx              # 评估页面（21.4 KB）
│   └── ComponentEvalPage.tsx     # 组件评估页面（20.1 KB）
├── components/
│   ├── Layout.tsx                # 侧边栏布局（3.1 KB）
│   ├── figma/
│   │   └── ImageWithFallback.tsx # 图片组件（受保护）
│   └── ui/                       # UI 组件库
│       ├── utils.ts              # ✅ 工具函数（必需）
│       ├── badge.tsx             # ✅ 标签
│       ├── button.tsx            # ✅ 按钮
│       ├── card.tsx              # ✅ 卡片
│       ├── collapsible.tsx       # ✅ 折叠面板
│       ├── dialog.tsx            # ✅ 对话框
│       ├── input.tsx             # ✅ 输入框
│       ├── label.tsx             # ✅ 标签文本
│       ├── scroll-area.tsx       # ✅ 滚动区域
│       ├── select.tsx            # ✅ 下拉选择
│       ├── slider.tsx            # ✅ 滑块
│       ├── sonner.tsx            # ✅ Toast 通知
│       ├── switch.tsx            # ✅ 开关
│       ├── table.tsx             # ✅ 表格
│       ├── tabs.tsx              # ✅ 标签页
│       └── textarea.tsx          # ✅ 多行文本框
└── utils/
    └── api.ts                    # API 封装（2.9 KB）
```

#### `/src/styles/` - 样式
```
src/styles/
├── fonts.css                     # 字体导入
└── theme.css                     # Tailwind 主题配置
```

---

## 📊 页面功能详解

### 1. Home.tsx (主页) ⭐
**路径**: `/`  
**功能**: RAG 检索增强生成  
**特点**:
- 现代 ChatBot 风格界面
- 流式对话体验
- 支持配置检索参数（top_k, collection, 模型等）
- 支持按 filepath/doc_id 过滤
- 支持 Hybrid Search（dense + sparse）
- 显示检索到的上下文

**使用的 API**:
- `POST /api/v1/retrieval/generate`

**使用的 UI 组件**:
- Button, Input, Label, Badge, Switch, ScrollArea, Select, Collapsible

---

### 2. ChunkingPage.tsx
**路径**: `/chunking`  
**功能**: 文本分块  
**特点**:
- 支持 4 种分块方法（Token, Semantic, LlamaIndex, Lumber）
- 直接文本输入 + 文件路径输入
- 实时显示分块结果
- 支持动态参数配置

**使用的 API**:
- `GET /api/v1/chunks/methods`
- `POST /api/v1/chunks/chunk-text`
- `POST /api/v1/chunks/chunk-file`

**使用的 UI 组件**:
- Card, Button, Input, Label, Textarea, Tabs, ScrollArea, Badge, Select

---

### 3. IndexPage.tsx
**路径**: `/index`  
**功能**: 向量索引管理  
**特点**:
- 构建/追加向量索引
- 列出所有 collections
- 查看 collection 详情（schema + 示例数据）
- 删除 collection
- 按 metadata 删除部分向量
- 支持稀疏向量（BM25）

**使用的 API**:
- `POST /api/v1/index/build`
- `POST /api/v1/index/add`
- `GET /api/v1/index/collections`
- `GET /api/v1/index/collections/inspect`
- `DELETE /api/v1/index/collections/{name}`
- `POST /api/v1/index/collections/{name}/delete-by-metadata`

**使用的 UI 组件**:
- Card, Button, Input, Label, Badge, Switch, Table, Dialog, Select, ScrollArea

---

### 4. RetrievalPage.tsx
**路径**: `/retrieval`  
**功能**: 向量检索（无生成）  
**特点**:
- 纯检索功能（不调用 LLM）
- 支持配置检索参数
- 支持文件批处理
- 显示检索结果和相似度分数

**使用的 API**:
- `POST /api/v1/retrieval/search`
- `POST /api/v1/retrieval/generate-file`

**使用的 UI 组件**:
- Card, Button, Input, Label, Textarea, Switch, Badge, Tabs, ScrollArea, Collapsible

---

### 5. EvalPage.tsx
**路径**: `/eval`  
**功能**: 端到端评估  
**特点**:
- Traditional Metrics（F1, ROUGE, BLEU, BERTScore）
- RAGAS 评估（7 个指标）
- 支持直接 JSON 输入（粘贴评估）
- 支持文件输入
- 自动识别格式（标准 RAGAS / sample_results.json）
- 详细的结果表格展示

**使用的 API**:
- `POST /api/v1/eval/traditional`
- `POST /api/v1/eval/traditional-file`
- `POST /api/v1/eval/ragas`
- `POST /api/v1/eval/ragas-file`

**使用的 UI 组件**:
- Card, Button, Input, Label, Textarea, Switch, Tabs, ScrollArea, Table

---

### 6. ComponentEvalPage.tsx
**路径**: `/component-eval`  
**功能**: 组件级评估（分块质量 + 黏连度）  
**特点**:
- Chunk Quality（BC + 语义不相似度）
- Chunk Stickiness（结构熵）
- **参数可视化调节**（threshold, delta 滑块）
- **力导向图**（块依赖关系网络）
- **相似度热力图**（N×N 矩阵）
- 实时参数调整 + 自动重新评估

**使用的 API**:
- `POST /api/v1/component-eval/chunk-quality`
- `POST /api/v1/component-eval/chunk-quality-file`
- `POST /api/v1/component-eval/chunk-stickiness`
- `POST /api/v1/component-eval/chunk-stickiness-file`

**使用的 UI 组件**:
- Card, Button, Input, Label, Textarea, Switch, Slider, Tabs, ScrollArea

**特殊依赖**:
- `react-force-graph-2d` - 力导向图可视化
- `recharts` - 图表展示

---

## 🎨 UI 组件使用统计

| 组件 | 使用次数 | 主要用途 |
|------|---------|---------|
| **Button** | 6 | 所有页面的操作按钮 |
| **Input** | 6 | 表单输入 |
| **Label** | 6 | 表单标签 |
| **Card** | 6 | 内容容器 |
| **ScrollArea** | 5 | 滚动区域 |
| **Tabs** | 4 | 标签页切换 |
| **Badge** | 4 | 状态标签 |
| **Switch** | 4 | 开关选项 |
| **Textarea** | 4 | 多行文本输入 |
| **Select** | 3 | 下拉选择 |
| **Collapsible** | 2 | 折叠展开 |
| **Table** | 2 | 数据表格 |
| **Dialog** | 1 | 对话框（IndexPage） |
| **Slider** | 1 | 参数滑块（ComponentEval） |
| **Sonner** | 1 | Toast 通知（全局） |

---

## 📦 核心依赖说明

### UI 框架
- `react` + `react-dom` - React 核心
- `react-router` - 路由管理

### UI 组件库
- `@radix-ui/react-*` - 无样式基础组件（只需安装实际使用的）
- `tailwindcss` - CSS 框架
- `lucide-react` - 图标库

### 数据可视化
- `recharts` - 图表库（用于评估结果趋势图）
- `react-force-graph-2d` - 力导向图（用于 Component Eval）

### 工具库
- `sonner` - Toast 通知
- `tailwind-merge` - Tailwind 类名合并
- `clsx` - 条件类名

---

## 🗑️ 可删除的文件（未使用）

### UI 组件（32 个）
```
src/app/components/ui/accordion.tsx
src/app/components/ui/alert-dialog.tsx
src/app/components/ui/alert.tsx
src/app/components/ui/aspect-ratio.tsx
src/app/components/ui/avatar.tsx
src/app/components/ui/breadcrumb.tsx
src/app/components/ui/calendar.tsx
src/app/components/ui/carousel.tsx
src/app/components/ui/chart.tsx
src/app/components/ui/checkbox.tsx
src/app/components/ui/command.tsx
src/app/components/ui/context-menu.tsx
src/app/components/ui/drawer.tsx
src/app/components/ui/dropdown-menu.tsx
src/app/components/ui/form.tsx
src/app/components/ui/hover-card.tsx
src/app/components/ui/input-otp.tsx
src/app/components/ui/menubar.tsx
src/app/components/ui/navigation-menu.tsx
src/app/components/ui/pagination.tsx
src/app/components/ui/popover.tsx
src/app/components/ui/progress.tsx
src/app/components/ui/radio-group.tsx
src/app/components/ui/resizable.tsx
src/app/components/ui/separator.tsx
src/app/components/ui/sheet.tsx
src/app/components/ui/sidebar.tsx
src/app/components/ui/skeleton.tsx
src/app/components/ui/toggle-group.tsx
src/app/components/ui/toggle.tsx
src/app/components/ui/tooltip.tsx
src/app/components/ui/use-mobile.ts
```

---

## 🔧 配置文件

### `vite.config.ts`
Vite 配置，需要保留

### `tsconfig.json`
TypeScript 配置，需要保留

### `index.html`
HTML 入口，需要保留

---

## 📝 迁移检查清单

- [ ] 复制 6 个页面文件
- [ ] 复制 Layout.tsx 组件
- [ ] 复制 16 个 UI 组件
- [ ] 复制 api.ts 工具
- [ ] 复制样式文件
- [ ] 修改 API_BASE_URL 为后端地址
- [ ] 安装核心依赖（参考 MINIMAL_PACKAGE.json）
- [ ] 测试所有页面功能
- [ ] 删除未使用的 32 个 UI 组件（可选）
- [ ] 清理 package.json 中未使用的依赖（可选）

---

## 🎯 总结

**核心文件**: 26 个  
- 6 个页面
- 2 个核心组件
- 16 个 UI 组件
- 1 个工具文件
- 1 个路由配置

**可删除**: 32 个未使用的 UI 组件

**项目精简度**: 删除后减少 ~40% 文件数量

准备迁移！🚀
