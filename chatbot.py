import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pprint import pprint
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True,
)

docs = loader.load()

MARKDOWN_SEPARATOR = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\**\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
#Split Documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=200,
#     add_start_index=True,
#     strip_whitespace=True,
#     separators=MARKDOWN_SEPARATOR,
# )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # Dùng bản small cho nhẹ, giống bên example.py
    base_url="https://openrouter.ai/api/v1",
    chunk_size=16, # QUAN TRỌNG: Chia nhỏ lô gửi đi để tránh bị OpenRouter chặn vì payload quá lớn
)

import os

FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True # Cần thiết cho phiên bản FAISS mới
    )
else:
    print("Creating new FAISS index and saving to disk...")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=0.85
    )

    splits = text_splitter.split_documents(docs)

    # BẮT BUỘC: Lọc bỏ các đoạn text rỗng để tránh lỗi "No embedding data received"
    splits = [s for s in splits if len(s.page_content.strip()) > 0]

    vectorstore = FAISS.from_documents(
        documents=splits, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    # Lưu lại để dùng cho lần sau
    vectorstore.save_local(FAISS_INDEX_PATH)

# Tăng k=10 để lấy được nhiều context hơn, bù đắp cho việc chunk có thể bị đứt gãy ngữ cảnh
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES: \n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say:"
    ' "I don\'t know based on the provided documents."\n'
    "3) Do NOT use outside knowledge bases, guessing, or wed information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context: \n{context}\n\n"
    "Question: \n{question}\n\n"
)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = input("Question: ")

answer = rag_chain.invoke(question)

pprint(answer)
