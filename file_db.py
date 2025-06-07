from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import datetime as datatime
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)


model = SentenceTransformer(r"E:/xf/rag/model_cache/BAAI/bge-large-zh-v1___5")

client = chromadb.PersistentClient("./chromadb")
collection = client.get_or_create_collection("my_collection",
                                        metadata={"hnsw:space": "cosine",
                                                  "创建时间":datatime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

def pdf2db2():
    try:
        # 1. 加载文档
        loader = PyPDFLoader(r"E:/xf/rag/使用说明书.pdf")
        pages = loader.load()
        if not pages:
            raise ValueError("PDF文档加载失败或为空")

        # 2. 文本分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。"]
        )
        chunks = splitter.split_documents(pages)
        
        # 3. 生成嵌入
        texts = [chunk.page_content for chunk in chunks]
        embeddings = model.encode(texts)
        
        if len(embeddings) == 0:
            raise ValueError("模型未生成有效嵌入")

        # 4. 存储到ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=[f"doc_{i}" for i in range(len(chunks))],
            metadatas=[chunk.metadata for chunk in chunks]
        )
        print(f"成功存储{len(chunks)}个文本块")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise

def pdf2db():
    loader = PyPDFLoader(r"E:/xf/rag/使用说明书.pdf")

    pages = loader.load()

    # 步骤2：递归分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。"]
    )
    chunks = splitter.split_documents(pages)

    # 步骤3：添加元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({"chunk_id": i, "doc_type": "technical_manual"})
    chunk_texts = [chunk.text for chunk in chunks if hasattr(chunk, 'text')]
    embeddings = model.encode(chunk_texts)
    # 步骤4：向量化存储（以FAISS为例）
   #from langchain.embeddings import OpenAIEmbeddings
   #vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    collection.add( embeddings=embeddings.tolist(),
                    documents=chunks,
                    ids=["doc_"+str(i) for i in range(len(chunks))],
                    metadatas=[{"filename": str(path)} for path in chunks])
    
    print(f"已将{len(chunks)}个文本转换为向量并存入数据库")

def txt2db():

    #读取当前目录下的txt文件赋值给text
    path = Path(r"E:/xf/rag/chromadb.txt")
    text = path.read_text(encoding='utf-8')
    #把text按行分割成列表
    text = text.splitlines()
    text_list = []
    for i, text1 in enumerate(text):
            text_list.append(text1)
            print(f"已读取第{i+1}行文本：{text1}")
#把文本转换为向量
    embeddings = model.encode(text_list)

#把向量转换为数据库
    collection.add( embeddings=embeddings.tolist(),
                    documents=text_list,
                    ids=["doc_"+str(i) for i in range(len(text_list))],
                    metadatas=[{"filename": str(path)} for path in text_list])
    
    print(f"已将{len(text_list)}个文本转换为向量并存入数据库")

def query_db(query):
    query_embedding = model.encode(query)
    results = collection.query(query_embedding.tolist(),n_results=1)
    return results["documents"][0]

if __name__ == "__main__":
    #txt2db()
    #df2db2()
    textes = query_db("遇到紧急情况该怎么办？")   
    for text in textes:
        print(text)
 

