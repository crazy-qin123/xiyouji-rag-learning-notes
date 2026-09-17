import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"   # ① 必须和 build_index 一致!
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章:"   # ② bge 检索前缀
TOP_K = 4

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),   # ⑤ key 从环境变量读,不写死在代码里!
    base_url="https://api.deepseek.com",          # ⑥ 换成 DeepSeek 的地址
    temperature=0.3,                              # ⑦ 0~2,越低回答越保守
)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        embedding_function=embeddings,     # ③ 注意!读库用的是 embedding_function
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def retrieve(vectorstore, question):
    hits = vectorstore.similarity_search_with_score(
        QUERY_INSTRUCTION + question,      # ④ 只给"问题"加前缀,文档不加
        k=TOP_K,
    )
    return hits

def build_prompt(question, hits):
    context = ""
    for i, (doc, score) in enumerate(hits, 1):
        context += f"【片段{i}|{doc.metadata['chapter']}】\n{doc.page_content}\n\n"
    prompt = f"""你是《西游记》知识问答助手。请只根据下面提供的原文片段回答问题,原文中没有的内容不要编造。

【原文片段】
{context}
【问题】
{question}
【回答】"""
    return prompt

def generate(prompt):
    response = llm.invoke(prompt)
    return response.content

def answer_one(vectorstore, question):
    hits = retrieve(vectorstore, question)
    prompt = build_prompt(question, hits)
    answer = generate(prompt)
    return answer, hits

def main():
    vectorstore = load_vectorstore()
    print("《西游记》问答助手启动!输入 exit 退出")
    while True:
        question = input("你问: ")
        if question.strip().lower() == "exit":
            break
        answer, hits = answer_one(vectorstore, question)
        print("\n回答:", answer)
        print()

if __name__ == "__main__":
    main()
