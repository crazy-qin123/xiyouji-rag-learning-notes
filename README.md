# 《西游记》RAG 问答实验

一个用于学习检索增强生成（RAG）完整流程的中文问答项目。项目将《西游记》文本按回目和语义片段切分，使用中文 Embedding 模型建立 Chroma 向量索引，再把检索到的原文片段交给 DeepSeek 生成有依据的回答。

## 已实现功能

- 从 Project Gutenberg 下载《西游记》文本
- 清理文本并尝试使用 OpenCC 转为简体中文
- 按回目识别章节，再以 500 字、80 字重叠进行递归分块
- 使用 `BAAI/bge-small-zh-v1.5` 生成归一化向量
- 将文本、向量和回目信息保存到 Chroma
- 为问题增加 BGE 检索指令并召回 Top-4 原文片段
- 通过兼容 OpenAI 接口的 DeepSeek 模型生成回答
- API Key 仅从环境变量读取

## 技术栈

Python、LangChain、Chroma、Hugging Face Embeddings、Sentence Transformers、OpenCC、DeepSeek API。

## 工作流程

```text
Project Gutenberg 原文
        ↓ 下载与清洗
按“第 X 回”划分章节
        ↓
递归文本分块
        ↓
BGE 中文向量化 → Chroma 持久化索引
        ↓
问题向量化 → Top-4 相似片段
        ↓
原文上下文 + 问题 → DeepSeek → 回答
```

## 运行方法

建议使用 Python 3.10 或更新版本，并创建独立虚拟环境。

```bash
pip install -r requirements.txt
python build_index.py
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="<your-api-key>"
python query_rag.py
```

macOS / Linux：

```bash
export DEEPSEEK_API_KEY="<your-api-key>"
python query_rag.py
```

首次运行 `build_index.py` 会下载原文和向量模型，并在本地生成 `novel/` 与 `chroma_db/`。这些可重建文件没有提交到仓库。

## 项目结构

```text
.
├── build_index.py
├── query_rag.py
├── requirements.txt
├── docs/
│   ├── 西游记RAG学习材料.md
│   └── build_index_note.md
└── README.md
```

## 我使用 Codex 的实际工作流

在这个学习项目中，我使用 Codex 辅助梳理 RAG 各阶段、解释函数和第三方库的职责、检查 API Key 的读取方式、整理公开仓库结构与学习文档。代码中的文本清洗、分块参数、检索数量和提示词仍为我自己理解、编写、运行、调整并解释。

## 当前状态

这是可以演示核心 RAG 链路的学习实验，已完成建索引和交互问答代码。当前仍属于命令行原型：尚未提供 Web 界面、自动化测试、检索评估集和生产部署方案。

更完整的逐函数学习记录见 [`docs/西游记RAG学习材料.md`](docs/西游记RAG学习材料.md)。

## 数据与版权

小说文本由脚本从 Project Gutenberg 下载，没有提交到本仓库。使用或再分发该文本时，请自行确认所在地区的版权状态以及 Project Gutenberg 的使用条款。项目源代码与原创文档采用 MIT License。
