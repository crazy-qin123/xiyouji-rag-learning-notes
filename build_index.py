import os
import urllib.request
import re

os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET","1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_URL = "https://www.gutenberg.org/ebooks/23962.txt.utf-8"
BOOK_PATH = os.path.join(BASE_DIR,"novel","xiyouji.txt")
CLEAN_PATH = os.path.join(BASE_DIR,"novel","xiyouji_clean.txt")
CHROMA_DIR = os.path.join(BASE_DIR,"chroma_db")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 500     #切块每块最多500字
CHUNK_OVERLAP = 80   #每块重复80字

def download_xiyouji() -> None:
    if os.path.exists(BOOK_PATH):
        print("文件已存在")
        return
    os.makedirs(os.path.dirname(BOOK_PATH),exist_ok=True)  #`os.path.dirname(BOOK_PATH)` —— 取路径的**文件夹部**,`os.makedirs(路径)` —— 递归创建目录，父目录不存在会一并创建 (`os.mkdir` 只能建一层，`makedirs` 能建整条链)。,`exist_ok=True` —— **关键字参数**(按名字传参): 目录已存在时不报错。注意这里的 `=` 不是赋值，是 "给参数指定值"。
    req = urllib.request.Request(BOOK_URL, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req,timeout=120) as resp:
        data = resp.read()
        with open(BOOK_PATH,"wb") as f:  #`"wb"` —— 模式字符串: `w` = write (写),`b` = binary (二进制), 合起来是 "以二进制写入模式打开"。文件不存在会自动创建，存在会**清空覆盖**
            f.write(data)



def load_and_clean() -> str:
    with open(BOOK_PATH,"r",encoding="utf_8",errors="ignore") as f:     #`errors="ignore"`个别字节解不了码就**丢弃**, 不让程序崩掉
        text = f.read()

    #只保留两条版权标记之间的正文
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[start:end]

    #翻译-简体
    try:
        import opencc
        try:
            convert = opencc.T2S()        #新版API
        except Exception:
            convert = opencc.OpenCC("t2s")    #旧版API
        text = convert.convert(text)
    except ImportError:
        print("！！！警告，未安装opencc-python-reimplemented,跳过繁体转简体！！！")

    with open(CLEAN_PATH,"w",encoding="utf-8") as f:
        f.write(text)

    return text

def split_by_chapters(text) -> list:
    pattern = re.compile(r"^\s*第[0-9〇零一二三四五六七八九十百千]+回[^\n]*",re.MULTILINE)
    matches = list(pattern.finditer(text))   #`pattern.finditer(text)`返回**所有匹配结果**的迭代器 (不是只找第一个)

    if len(matches) < 2:
        return [("未知回目",text)]    #如果没切出回目就当一整回

    chapters = []
    for i,m in enumerate(matches):
        title = m.group().strip()
        body_start = m.end()
        body_end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        chapters.append((title,body))
    return chapters

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_chunks(chapters) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       #500
        chunk_overlap=CHUNK_OVERLAP,   #80
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]         #尽量把句子落在完整的地方
    )

    docs = []
    for title,body in chapters:
        for piece in splitter.split_text(body):
            piece = piece.strip()
            if piece:
                docs.append(Document(page_content=piece,metadata={"chapter":title}))
    return  docs

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_index(docs) -> None:
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings":True},
    )

    vectorstores = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:sync_threshold": 5000},   # 绕开 chromadb 1.5.x 的 HNSW 损坏 bug
    )
    print("向量库已创建，共",vectorstores._collection.count(),"条")

if __name__ == "__main__":
    download_xiyouji()        # ① 下载
    text = load_and_clean()   # ② 清洗
    chapters = split_by_chapters(text)   # ③ 切回目
    docs = make_chunks(chapters)         # ④ 分块
    build_index(docs)