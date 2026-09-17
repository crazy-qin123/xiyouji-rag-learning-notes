# 《西游记》RAG 问答项目 · 学习材料

> 配套代码：[`build_index.py`](../build_index.py) 与 [`query_rag.py`](../query_rag.py)
> 面向对象:想搞懂"RAG 到底怎么跑起来"的初学者
> 阅读建议:先看第 1~3 章建立整体感觉,再对照代码逐函数读第 4、5 章,最后用第 6 章的问题自测

---

## 目录

1. [这个项目到底做了什么?](#1-这个项目到底做了什么)
2. [RAG 核心概念(大白话版)](#2-rag-核心概念大白话版)
3. [项目文件与全流程总览](#3-项目文件与全流程总览)
4. [build_index.py:建索引(离线阶段)函数逐个拆解](#4-build_indexpy-建索引离线阶段函数逐个拆解)
5. [query_rag.py:问答(在线阶段)函数逐个拆解](#5-query_ragpy-问答在线阶段函数逐个拆解)
6. [关键知识点深入:向量、距离、检索、Prompt](#6-关键知识点深入向量距离检索prompt)
7. [全部函数速查表](#7-全部函数速查表)
8. [第三方库分工一览](#8-第三方库分工一览)
9. [怎么运行](#9-怎么运行)
10. [常见问题排查](#10-常见问题排查)
11. [安全提醒](#11-安全提醒)
12. [扩展练习(动手加深理解)](#12-扩展练习动手加深理解)

---

## 1. 这个项目到底做了什么?

一句话:**把整本《西游记》变成"可检索的资料库",然后让大模型(DeepSeek)看着书里的原文回答你的问题。**

传统的直接提问:你问大模型"孙悟空的金箍棒哪来的?",它靠训练时的记忆回答——可能记错、可能编造。

本项目(RAG)的做法:先从《西游记》原文里**找出最相关的那几段**(检索),把原文片段塞给大模型,再让它回答——**回答有出处、不瞎编**。

代码只有两个文件,分工明确:

| 文件 | 阶段 | 干的事 |
|---|---|---|
| `build_index.py` | **离线准备**(跑一次即可) | 下载小说 → 清洗 → 切成小块 → 每块转成向量 → 存进向量库 |
| `query_rag.py` | **在线问答**(每次提问都跑) | 问题转向量 → 在向量库里找最像的 Top-4 片段 → 拼 Prompt → 调 DeepSeek 生成回答 |

---

## 2. RAG 核心概念(大白话版)

| 概念 | 英文/别名 | 大白话解释 | 类比 |
|---|---|---|---|
| RAG | Retrieval-Augmented Generation,检索增强生成 | 先"查资料"再"写答案"的问答方式 | 开卷考试:先翻书找到相关段落,再照着写 |
| 分块 | chunk / splitting | 把整本书切成几百字的小段 | 把一整本书拆成一页页小卡片 |
| 向量 / 向量化 | embedding | 把一段文字变成一串数字(如 512 个数),语义相近的文字,数字也相近 | 给每句话算一个"语义指纹" |
| 向量库 | vector store | 存放"文字+它的向量"的数据库,支持按向量相似度查找 | 一摞按语义排好序的卡片盒 |
| 相似度匹配 | retrieval / similarity search | 拿问题的向量,和库里所有向量比距离,取最近的几个 | 查字典:找你最像要找的那几页 |
| Top-K | top-k | 只取最相似的前 K 个片段 | 只翻最相关的 4 页,不翻整本书 |
| Prompt | 提示词 | 发给大模型的一段话(含指令和资料) | 给开卷考生的"考题+参考资料" |
| 幻觉 | hallucination | 大模型不懂装懂、编造内容 | 考生没翻书就乱写 |
| 上下文窗口 | context window | 大模型一次能"看到"的文字上限 | 考卷上能印多少参考资料 |

**为什么向量能匹配?** 向量模型把文字编码成坐标点。意思越接近的两句话,它们在多维空间里的坐标就越靠近。比如:
- "孙悟空的金箍棒是从哪里来的?" 的向量
- "悟空到龙宫寻宝,龙王献上如意金箍棒……" 的向量
两者距离很近,所以能被检索到;
而"唐僧师徒来到女儿国"的向量就离得很远,不会被当作答案来源。

---

## 3. 项目文件与全流程总览

### 3.1 目录结构

```
xiyouji-rag/
├── build_index.py           # 脚本①:建索引(本材料第 4 章)
├── query_rag.py             # 脚本②:问答(本材料第 5 章)
├── requirements.txt         # 依赖清单(pip 安装用)
├── README.md                # 简要使用说明
├── novel\
│   ├── xiyouji.txt          # 从古登堡计划下载的《西游记》原文(繁体,约 2.26 MB)
│   └── xiyouji_clean.txt    # 清洗后文本(简体,约 75.8 万字)—— 中间产物
└── chroma_db\               # 向量库(建好的"卡片盒")
    └── chroma.sqlite3       # 实际落盘的索引数据(1920 个向量)
```

### 3.2 一张图看懂全部流程

```
【build_index.py 离线阶段(第 1 步,跑一次)】
  下载小说 xiyouji.txt(繁体)
       │  download_book()
       ▼
  清洗:去版权样板 + 繁体转简体 → xiyouji_clean.txt
       │  load_and_clean()
       ▼
  按"第X回"切出 90 回
       │  split_by_chapters()
       ▼
  每回再切成 ~500 字的小块 → 1920 个 chunk
       │  make_chunks()
       ▼
  每个 chunk 用向量模型转成 512 维向量
       │  build_index() 里 HuggingFaceEmbeddings
       ▼
  全部写入 Chroma → chroma_db/(1920 条)
       │
       ▼  (到这里资料库就建好了,下面随时可问)

【query_rag.py 在线阶段(第 2 步,每次提问)】
  你输入问题"孙悟空的金箍棒是从哪里来的?"
       │
       ▼
  ① 用【同一个】向量模型把问题转成向量(前面加 bge 检索指令前缀)
       │  retrieve() 内部 embed_query
       ▼
  ② 在 chroma_db 里做相似度匹配,取距离最小的 Top-4 片段
       │  retrieve() 内部 similarity_search_with_score
       ▼
  ③ 把 4 段原文 + 答题纪律 拼成 Prompt
       │  build_prompt()
       ▼
  ④ 发给 DeepSeek 大模型生成回答
       │  generate()
       ▼
  ⑤ 打印回答 + 参考片段(便于核对出处)
```

> 两个文件之间唯一的"接口"就是 `chroma_db/` 目录:①建好,②直接用;
> 以及一个"暗号"——**两边必须用同一个向量模型**(`BAAI/bge-small-zh-v1.5`),
> 否则问题的向量和资料的向量不在同一个坐标系里,匹配会完全失效。

---

## 4. build_index.py:建索引(离线阶段)函数逐个拆解

### 4.0 全局配置区(不在函数里,但很重要)

| 代码 | 作用 | 白话解释 |
|---|---|---|
| `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")` | 把 HuggingFace 下载源换成国内镜像 | huggingface.co 在国内连不上,走镜像站才能下载模型 |
| `os.environ.setdefault("HF_HUB_DISABLE_XET", "1")` | 关闭新版 huggingface 的 xet 传输协议 | 该协议可能绕过镜像,关掉保证走 https 镜像 |
| `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` | 取脚本所在目录作为项目根目录 | 无论你在哪个目录执行 `python build_index.py`,文件路径都不会错 |
| `BOOK_PATH / CLEAN_PATH / CHROMA_DIR` | 三个关键文件路径 | 原文 / 清洗后文本 / 向量库,都放在项目目录下 |
| `EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"` | 向量模型名 | 中文效果不错且只有约 95MB,首次运行自动下载 |
| `CHUNK_SIZE = 500` | 每块最大 500 字 | 块太小语义不完整,块太大检索不精准 |
| `CHUNK_OVERLAP = 80` | 相邻块重叠 80 字 | 防止一句话恰好被从中间切断、两边都不完整 |

### 4.1 `download_book()` —— ① 下载小说

```python
def download_book() -> None:
    if os.path.exists(BOOK_PATH):        # 文件已存在就直接跳过
        print(...)
        return
    os.makedirs(os.path.dirname(BOOK_PATH), exist_ok=True)   # 建目录(已存在不报错)
    req = urllib.request.Request(BOOK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(BOOK_PATH, "wb") as f:
        f.write(data)
```

| 要点 | 说明 |
|---|---|
| 幂等(可重复执行) | 先检查文件在不在,在就直接 return,不会重复下载 |
| `User-Agent` 请求头 | 伪装成浏览器,很多网站会拒绝"裸奔"的脚本请求 |
| `urllib.request` | Python 标准库的 HTTP 客户端,不依赖第三方包 |
| `with ... as` 资源管理 | 用完自动关闭网络连接和文件,防止句柄泄漏 |

**它是怎么实现"下载"的?** 用标准库发一个 HTTP GET 请求到古登堡计划的文本地址,把返回的字节流(`resp.read()`)原样写到本地 `xiyouji.txt`。古登堡计划是公共版权书库,《西游记》在这里是繁体 TXT。

### 4.2 `load_and_clean()` —— ② 清洗文本

```python
def load_and_clean() -> str:
    with open(BOOK_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # 只保留两条版权标记之间的正文
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end   = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[start:end]
    # 繁体 -> 简体(opencc)
    try:
        import opencc
        try:
            converter = opencc.T2S()          # 新版 API
        except Exception:
            converter = opencc.OpenCC("t2s")  # 旧版 API
        text = converter.convert(text)
    except ImportError:
        print("[警告] 未安装 opencc-python-reimplemented,跳过繁体转简体。")
    # 落盘副本,方便检查
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text
```

**它清洗了什么?为什么要清洗?**

1. **去掉版权样板**:古登堡下载的 TXT 前后有几十行英文版权声明。用两个固定标记字符串定位正文起点和终点,`text[start:end]` 一下就把正文裁出来了。
2. **繁体转简体**:古登堡版《西游记》是繁体。向量模型主要学简体语料,而且你提问多半是简体,**繁体资料 + 简体问题匹配效果差**。opencc 是开源简繁转换库,一行 `converter.convert(text)` 全文转换。

| 细节 | 为什么 |
|---|---|
| `errors="ignore"` | 个别字节解不了码就丢弃,不让整个程序崩掉 |
| `find()` 返回 -1 判断 | 标记万一不存在,`text[start:end]` 会切错,所以先判 `!= -1` |
| try/except 兼容新旧 API | 不同版本的 opencc 类名不同,两个都试一遍 |
| 返回 text 且另存一份 | 返回给内存继续用;存盘后下次运行直接读,不用重复转换 |

### 4.3 `split_by_chapters(text)` —— ③a 按"第X回"切章节

```python
pattern = re.compile(r"^\s*第[0-9〇零一二三四五六七八九十百千]+回[^\n]*", re.MULTILINE)
matches = list(pattern.finditer(text))
if len(matches) < 2:
    return [("未知回目", text)]        # 兜底:没切出回目就把全书当一章
chapters = []
for i, m in enumerate(matches):
    title = m.group().strip()          # "第一回 灵根育孕源流出..."
    body_start = m.end()               # 本回正文从标题行之后开始
    body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[body_start:body_end]   # 到下一回标题为止
    chapters.append((title, body))
return chapters
```

**正则怎么读?**(这是一行"扫描仪")

```
^                行首
\s*              行首可以有空白(有些回目行前带空格)
第 ... 回        字面量"第"和"回"之间的部分匹配:
[0-9〇零一二三四五六七八九十百千]+   数字:阿拉伯数字或中文数字都行
[^\n]*           标题剩余内容(直到换行)
re.MULTILINE     让 ^ 能匹配"每一行"的开头(默认只匹配整个字符串开头)
```

**它怎么切?** 找齐所有回目标题行后,第 i 个标题到第 i+1 个标题之间的内容就是第 i 回的正文,用切片 `text[start:end]` 取出,得到 `[(回目标题, 正文), ...]` 这样的列表——相当于先把书按"章"拆好,下一步再细切。

> 说明:古登堡这个校对本的编号写成"第一一回"这类格式、缺整十回标题,所以实际切出 90 回而不是 100 回,正文内容完整,不影响问答。

### 4.4 `make_chunks(chapters)` —— ③b 每回再切成小块

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,      # 500
    chunk_overlap=CHUNK_OVERLAP,# 80
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
docs = []
for title, body in chapters:
    for piece in splitter.split_text(body):
        piece = piece.strip()
        if piece:
            docs.append(Document(page_content=piece, metadata={"chapter": title}))
```

| 对象 | 是什么 | 在本项目的意义 |
|---|---|---|
| `Document` | langchain 对"一段文本"的封装,含 `page_content`(正文)+ `metadata`(字典,附加信息) | 正文后续要向量化;`metadata={"chapter": title}` 记下"出自第几回",检索后能展示出处 |
| `RecursiveCharacterTextSplitter` | 递归字符切分器 | 把一回想切成几段 500 字的小块 |

**切分器的"递归"逻辑**:按 `separators` 的**先后顺序**尝试:
先按空行(`\n\n`)切 → 还有超长的段就按单换行切 → 再不行按句号、叹号、问号、分号、逗号、空格切 → 最后按字符硬切。
一句话:**尽量把断点落在语义完整的位置**(句子结尾),实在没办法才硬切。`chunk_overlap=80` 让相邻两块重叠 80 字,被切断的语义前后各保留一点。

**产出**:1920 个 `Document`,每个约 500 字,都带着"出自第几回"的标签。

### 4.5 `build_index(docs)` —— ④ 向量化 + ⑤ 入库(本项目最核心)

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},                    # 用 CPU 推理
    encode_kwargs={"normalize_embeddings": True},      # 向量归一化
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,                      # 落盘到 chroma_db/
)
```

**第 ④ 步:文字 → 向量,是怎么发生的?**

`HuggingFaceEmbeddings` 是 langchain 对 `sentence-transformers` 的封装。当你调用它的 `embed_documents(文本列表)` 时,底层流程是:
1. 用 tokenizer 把文字切成模型认识的最小单元(token);
2. 送入 Transformer 神经网络(bge-small-zh-v1.5,24M 参数);
3. 网络输出一个 512 维的向量,作为整段文字的"语义浓缩"。

关键参数:
- `device="cpu"`:你机器没配 GPU,让模型用 CPU 算;
- `normalize_embeddings=True`:把向量长度归一化为 1。归一化后,**两个向量的余弦相似度(夹角)等价于点积(数值相乘再求和)**,数学上更好排序、更稳。

**第 ⑤ 步:向量 + 文本 + 出处,怎么入库?**

`Chroma.from_documents(documents, embedding, persist_directory)` 内部替你做了三件事:
1. 对每个 `Document`,调用 `embeddings.embed_documents()` 批量转成向量;
2. 把三元组 `(向量, 文本内容, metadata)` 存进 Chroma;
3. 按 `persist_directory` 把数据写到磁盘 `chroma_db/` 下的 SQLite 文件,下次直接读盘,不用重新向量化整本书。

**为什么叫"索引"?** 就像书的目录告诉你"某某在第几页",向量索引让计算机不用把 1920 段全看一遍就能快速定位"哪几段和问题最接近"。

### 4.6 `main()` + 程序入口

```python
def main() -> None:
    if os.path.exists(CLEAN_PATH):      # 已清洗过就直接复用
        with open(CLEAN_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    else:                               # 第一次跑:下载 + 清洗
        download_book()
        text = load_and_clean()
    docs = make_chunks(split_by_chapters(text))
    build_index(docs)

if __name__ == "__main__":
    main()
```

| 要点 | 说明 |
|---|---|
| 复用 `xiyouji_clean.txt` | 清洗(opencc)很耗时,清洗过一次就存盘;想强制重洗,删掉该文件再跑即可 |
| `if __name__ == "__main__"` | 只有"直接运行本文件"时才执行 `main()`;被别人 `import` 时不执行。这是 Python 的标准入口写法,便于复用函数 |

---

## 5. query_rag.py:问答(在线阶段)函数逐个拆解

### 5.0 全局配置区

| 常量 | 值 | 含义 |
|---|---|---|
| `EMBEDDING_MODEL` | 与 build_index 相同 | **必须一致**,否则向量"坐标系"不同,匹配全废 |
| `QUERY_INSTRUCTION` | "为这个句子生成表示以用于检索相关文章:" | bge 官方建议:检索时给问题加指令前缀,模型会按"检索匹配"模式编码,召回更好。**只加给问题,文档不加** |
| `DEEPSEEK_API_KEY` | 环境变量(见第 11 章安全提醒) | DeepSeek 的密钥,不能写死在代码里 |
| `DEEPSEEK_BASE_URL` | https://api.deepseek.com | DeepSeek 服务器地址。它兼容 OpenAI 协议,所以能直接用 `ChatOpenAI` |
| `DEEPSEEK_MODEL` | deepseek-chat | DeepSeek 的对话模型;换 deepseek-reasoner 可做深度思考 |
| `TOP_K = 4` | 4 | 每次取最相似的 4 段喂给大模型 |

### 5.1 `load_vectorstore()` —— 把"卡片盒"打开

```python
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, ...)  # 同一个模型
vectorstore = Chroma(
    embedding_function=embeddings,   # 检索时用它给问题转向量
    persist_directory=CHROMA_DIR,    # 打开磁盘上已有的库
)
return vectorstore
```

与建索引对称:建索引时用 `Chroma.from_documents` **写**库,这里用 `Chroma(...)` **读**库(模型已缓存,加载只要几秒)。

### 5.2 `retrieve(vectorstore, question)` —— ③ 检索(向量匹配)

```python
hits = vectorstore.similarity_search_with_score(QUERY_INSTRUCTION + question, k=TOP_K)
```

**这一行就是整个 RAG 的"R"(Retrieval)。它内部做了两件事:**

1. **问题转向量**:调用 `embeddings.embed_query(加了前缀的问题)`,把一句话压成 512 维向量;
2. **相似度匹配**:拿问题向量和库里 **1920 个** chunk 向量逐一算距离(Chroma 默认 L2 欧氏距离,即"多维空间里的直线长度"),按距离从小到大排序,返回前 `TOP_K` 个 `(Document, score)`。

**score 怎么看?** score 是距离,**越小越相似**。例如检索"金箍棒从哪来"得到:
```
#1 [第五十二回] score=0.79 | 却说行者收了金箍棒...
#2 [第三回]     score=0.85 | ...龙宫...如意金箍棒...
```
数字小的是最相关的。这里还能看到 `metadata` 的用处:每个片段都标着出自第几回,方便人核对、方便大模型引用出处。

> 底层原理:Chroma 用的是 ANN(近似最近邻)索引,不会真的逐条和 1920 个向量比(那样慢),而是用数据结构(如 HNSW)快速缩小范围。对学习来说,理解"找距离最近的几个"即可。

### 5.3 `build_prompt(question, hits)` —— ④ 拼 Prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的中文助手。请只依据...原文片段回答...不要编造。"),
    ("human",  "以下是《西游记》原文片段:\n\n{context}\n\n用户问题:{question}\n\n..."),
])
messages = prompt.format_messages(context="\n\n".join(context_parts), question=question)
```

**Prompt = 给大模型的"考题 + 参考资料 + 答题纪律"。**

- **system(系统消息)**:设定身份与规则。这里的关键话术是"只依据片段回答、不足以回答就明说、不要编造"——这是在**防幻觉**;
- **human(用户消息)**:模板里 `{context}`、`{question}` 是占位符。`format_messages` 把检索到的 4 段原文(每段前面标了"【片段N·出自第X回】")和用户问题填进去,生成真正要发给模型的消息列表。

**为什么不直接把原文拼一段话发过去?** 用模板的好处是:结构固定、内容可替换、system/human 角色分明,这是 langchain 推荐的做法,也好维护。

### 5.4 `generate(question, messages)` —— ⑤ 调用大模型

```python
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,   # 指向 DeepSeek,而不是 OpenAI
    temperature=0.3,              # 回答更稳、更贴原文
    timeout=60,
)
answer = llm.invoke(messages).content
```

**这是 RAG 的"G"(Generation)。**

- `ChatOpenAI` 本是为 OpenAI 写的封装;DeepSeek 对外提供 OpenAI 兼容接口,所以**只改 `base_url` 和 `model` 就能无缝切换**,其余代码不用动——这是 OpenAI 协议成为"事实标准"带来的便利;
- `temperature=0.3`:控制随机性。越低回答越稳定、越忠实于原文;想更有创造性就调高(0~2);
- `invoke(messages)`:向 `https://api.deepseek.com/chat/completions` 发一次 HTTP 请求,返回 AIMessage,取 `.content` 就是模型写的回答文本。

### 5.5 `answer_one(vectorstore, question)` —— 串联 ②③④⑤

```python
hits = retrieve(vectorstore, question.strip())   # 检索
messages = build_prompt(question.strip(), hits)  # 拼 Prompt
answer = generate(question.strip(), messages)    # 生成
print(f">>> 回答:\n{answer}")
# 再把命中的原文片段打印出来供人工核对
```

这个函数把"一个问题"走完整个链路,并负责打印"回答"与"参考片段"(参考片段生产环境一般不给终端用户看,这里是为了让你能验证回答确实有出处)。若 `generate` 返回 None(没配 key),提前 return,不打印假回答。

### 5.6 `main()` —— 命令行入口

```python
parser.add_argument("question", nargs="?", ...)   # 可选:要问的问题
parser.add_argument("-i", "--interactive", ...)   # 可选:交互模式
if not os.path.isdir(CHROMA_DIR):                 # 前置检查
    print("请先运行 python build_index.py ...")
    return
vectorstore = load_vectorstore()                  # 只加载一次
if args.interactive or not args.question:         # 交互循环
    while True:
        q = input("请输入问题: ").strip()
        ...
        answer_one(vectorstore, q)
else:
    answer_one(vectorstore, args.question)
```

| 设计 | 原因 |
|---|---|
| `argparse` | 解析命令行:`python query_rag.py "问题"` 或 `-i` 进交互模式 |
| 前置检查 `os.path.isdir(CHROMA_DIR)` | 没建过索引就提示先跑 build_index,而不是抛底层异常 |
| `vectorstore` 只加载一次 | 交互模式下问 10 个问题不用重载 10 次模型 |
| `input()` + try/except | `Ctrl+C`、输入 `exit` 都能优雅退出 |

---

## 6. 关键知识点深入:向量、距离、检索、Prompt

### 6.1 向量到底是什么?

一段文字 → 向量模型 → 一串数字,例如 512 个数:
```
"孙悟空到龙宫寻宝"  → [0.023, -0.157, 0.089, ..., 0.311]   (512 个数)
"唐僧被妖怪抓走"    → [-0.201, 0.077, -0.133, ..., -0.245]
```
这串数字可以看作**512 维空间里的一个点**。模型在训练时学到:语义相关的句子,它们的点靠得近。

### 6.2 怎么算"靠得近"?两种常见距离

- **欧氏距离(L2)**:两点之间直线长度。`score` 越小越相似。
- **余弦相似度**:看两个向量方向的夹角,cos 越接近 1 越相似。
- 本项目两个技巧:`normalize_embeddings=True`(向量长度归一化)后,L2 距离和余弦相似度排序结果等价,计算更简单。

### 6.3 为什么"检索"比"关键词搜索"强?

关键词搜索只能匹配"字面相同"(搜"金箍棒"找不到"如意棒")。向量检索匹配的是**语义**:
- 问"孙悟空被压在哪座山?" → 原文写的是"五行山下定心猿" → 字面完全不同,但语义相同,向量距离很近,能被召回。
这就是 RAG 能"读懂意思"去翻书的原因。

### 6.4 Prompt 的结构(看得见的"魔法")

发给 DeepSeek 的最终消息长这样(简化):

```
system: 你是严谨的中文助手,只依据《西游记》原文片段回答;不足以回答就说"无法回答",不要编造。

user:   以下是《西游记》原文片段:
        【片段1·出自第五十二回】
        却说行者收了金箍棒...
        【片段2·出自第三回】
        ...龙王无奈献出如意金箍棒...

        用户问题:孙悟空的金箍棒是从哪里来的?

        请结合片段给出简洁、有依据的回答。
```

模型"看着"这些原文写答案,自然就有依据、不跑偏。

### 6.5 防幻觉的三道防线(本项目用了哪些)

1. **检索限定来源**(最重要):只把命中的原文给模型,答案必须基于它;
2. **Prompt 纪律**:system 里明确"不足以回答就说无法回答,不要编造";
3. **展示参考片段**:人工可以核对回答是否有出处。

---

## 7. 全部函数速查表

### build_index.py(6 个函数)

| 函数 | 输入 | 输出 | 作用 | 调用关系 |
|---|---|---|---|---|
| `download_book()` | 无 | None | 下载《西游记》到 novel/xiyouji.txt(已存在则跳过) | 被 `main()` 调用 |
| `load_and_clean()` | 无 | str(清洗后全文) | 去版权样板 + 繁体转简体,并存盘 clean 文件 | 被 `main()` 调用 |
| `split_by_chapters(text)` | str 全文 | list[(回目标题, 正文)] | 正则定位"第X回",切成 90 章 | 被 `main()` 调用 |
| `make_chunks(chapters)` | 章节列表 | list[Document] | 每章再切成 ~500 字小块,附 chapter 元数据 | 被 `main()` 调用 |
| `build_index(docs)` | list[Document] | None | 向量化 + 写入 Chroma 并持久化 | 被 `main()` 调用 |
| `main()` | 无 | None | 组装:清洗(或复用)→ 分块 → 建索引 | 程序入口 |

### query_rag.py(6 个函数)

| 函数 | 输入 | 输出 | 作用 | 调用关系 |
|---|---|---|---|---|
| `load_vectorstore()` | 无 | Chroma 对象 | 加载向量模型 + 打开磁盘向量库 | 被 `main()` 调用 |
| `retrieve(vectorstore, question)` | 向量库,问题 | list[(Document, score)] | 问题转向量 → 相似度匹配 → 取 Top-4 | 被 `answer_one` 调用 |
| `build_prompt(question, hits)` | 问题,检索结果 | list[messages] | 把片段拼成 system/human 消息 | 被 `answer_one` 调用 |
| `generate(question, messages)` | 问题,消息 | str 或 None | 调 DeepSeek 生成回答(没 key 返回 None) | 被 `answer_one` 调用 |
| `answer_one(vectorstore, question)` | 向量库,问题 | None | 串联 检索→拼 Prompt→生成→打印 | 被 `main()` 调用 |
| `main()` | 无 | None | 命令行解析、前置检查、单问/交互模式 | 程序入口 |

### 语言内置函数/方法也用到了这些(认识一下)

`open()`、`os.path.exists`、`os.makedirs`、`os.path.dirname`、`os.path.abspath`、`re.compile`、`re.MULTILINE`、`str.find`、`str.strip`、`enumerate`、`input`、`argparse` 等——都是 Python 自带能力,不需要装包。

---

## 8. 第三方库分工一览

| 库 | 在本项目扮演的角色 | 关键类/函数 |
|---|---|---|
| `langchain-core` | 数据与提示词基础 | `Document`(文本+元数据)、`ChatPromptTemplate`(提示词模板) |
| `langchain-text-splitters` | 分块 | `RecursiveCharacterTextSplitter` |
| `langchain-huggingface` | 把本地向量模型包装成 langchain 接口 | `HuggingFaceEmbeddings` |
| `langchain-chroma` | 让 langchain 操作 Chroma 向量库 | `Chroma` |
| `langchain-openai` | 通过 OpenAI 协议调 DeepSeek | `ChatOpenAI` |
| `sentence-transformers` | 真正干向量化的本地模型库 | 加载 `BAAI/bge-small-zh-v1.5` |
| `chromadb` | 向量数据库(存储+近似检索) | PersistentClient / Collection |
| `torch` | Transformer 模型的计算引擎(CPU 版) | — |
| `transformers` | 模型/tokenizer 加载与推理 | — |
| `opencc-python-reimplemented` | 繁体转简体 | `opencc.OpenCC("t2s")` |
| `openai`(间接) | ChatOpenAI 底层 HTTP 客户端 | — |

> 全部在 `requirements.txt`;注意 Windows 下 `pip install -r` 要求该文件是纯 ASCII/UTF-8 且无编码问题,中文注释可能让 pip 按 GBK 读取报 `UnicodeDecodeError`——所以本项目 requirements.txt 用英文注释。

---

## 9. 怎么运行

```bat
cd xiyouji-rag

REM ①(可选)重建索引:下载→清洗→切块→向量化→入库,约几分钟
python build_index.py

REM ②配置 key(若代码未内置;详见第 11 章)
set DEEPSEEK_API_KEY=sk-你的key

REM ③单次提问
python query_rag.py "孙悟空的金箍棒是从哪里来的?"

REM ④交互模式(输入 exit 退出)
python query_rag.py -i
```

索引已经建好(`chroma_db\` 已存在),通常你只需执行 ③。

---

## 10. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain_chroma'` | 用错了 Python 环境 | 用 `python`(已装依赖)或 anaconda base 的 `python` |
| 提示"未检测到 DEEPSEEK_API_KEY" | 没配 key / 环境变量丢了 | `set DEEPSEEK_API_KEY=sk-你的key`(单独一行,别跟其他命令) |
| 输出乱码 | cmd 代码页不是 UTF-8 | 先执行 `chcp 65001` 再运行 |
| 检索结果不相关 | 换过向量模型 / 问题太笼统 | 两边模型必须一致;把问题问具体;调大 `TOP_K` |
| 回答胡说八道 | 片段不够或 Prompt 纪律弱 | 缩小 `CHUNK_SIZE` 或调大 `TOP_K`;检查参考片段是否真的相关 |
| `pip install -r` 报编码错误 | requirements.txt 含中文被按 GBK 读 | 注释用英文,或转成 UTF-8 with BOM |

---

## 11. 安全提醒 ⚠️

**API Key 属于密码,绝不能写死在代码里**,否则:
- 代码一旦上传 GitHub / 发给别人,key 立即泄露;
- 别人可以用你的 key 调用 DeepSeek,费用算在你头上。

请把 key 改回从环境变量读取的方式(注释里也这么写了):

```python
# 正确:从环境变量读取,默认空
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# 错误示例:不要把真实 Key 直接写入代码
# DEEPSEEK_API_KEY = "<your-api-key>"
```

然后运行前在 cmd 设置:
```bat
set DEEPSEEK_API_KEY=sk-你的key
```
(或 `setx DEEPSEEK_API_KEY "sk-你的key"` 永久保存,需重开 cmd。)

> 如果你的 key 已经在代码里出现过,且这段代码可能被分享过,建议去 DeepSeek 平台**删除并重新生成**该 key。

---

## 12. 扩展练习(动手加深理解)

1. **改参数看效果**:把 `query_rag.py` 的 `TOP_K` 改成 1 和 8 各问一次,感受"上下文太少/太多"对回答的影响。
2. **换问题类型**:问事实型("唐僧有几个徒弟?")、情节型("女儿国国王为什么放走唐僧?"),观察召回片段和回答质量差异。
3. **换向量模型**:把 `EMBEDDING_MODEL` 换成 `BAAI/bge-large-zh-v1.5`,重建索引再对比检索分数(注意:两处要一起换)。
4. **换一本书**:改 `BOOK_URL` 与正则(如"第X章"),跑一遍 build_index,就能对任何公共版权小说提问。
5. **手动算一遍相似度**:用 numpy 自己实现"问题向量 × 库向量 → 排序取 Top-4",和 `similarity_search_with_score` 的结果对比,彻底搞懂向量匹配。
6. **看看 chroma.sqlite3**:用 DB Browser 打开 `chroma_db\chroma.sqlite3`,观察向量、文本、metadata 是怎么存的。
7. **去掉防幻觉话术**:把 system 提示词改成"自由发挥",对比回答,直观感受幻觉防护的作用。

---

*材料制作日期:2026-09-08,对应 `build_index.py` / `query_rag.py` 的当前版本。*



