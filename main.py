import os
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
import asyncio
import google.generativeai as genai
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rsge-rag-final")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="RS.ge Ultimate RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


embeddings = None
vector_db = None
bm25_retriever = None
reranker = None

class Question(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    global embeddings, vector_db, bm25_retriever, reranker
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"მოდელი ეშვება: {device.upper()} რეჟიმში")
    
    logger.info("მოდელების ჩატვირთვა...")

    
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

 
    collection = vector_db._collection
    total = collection.count()
    batch_size = 5000
    all_docs = []
    
    for i in range(0, total, batch_size):
        batch = collection.get(limit=batch_size, offset=i, include=["documents", "metadatas"])
        for txt, meta in zip(batch['documents'], batch['metadatas']):
            if txt:
                all_docs.append(Document(page_content=txt, metadata=meta or {}))
    
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=60)
    logger.info(f"სისტემა მზადაა. ბაზაშია {total} დოკუმენტი.")

def get_context(query: str):
    
    case_num_match = re.search(r"(\d{4,6})", query)
    
    if case_num_match:
        case_id = case_num_match.group(1)
        logger.info(f"აღვადგენ მთლიან დოკუმენტს N {case_id}...")
        
        results = vector_db.get(where={"case_number": case_id})
        
        if results and results['documents']:
            
            full_text = "\n\n".join(results['documents'])
            return [Document(page_content=full_text, metadata={"case_number": case_id})]

    
    dense_hits = vector_db.similarity_search(query, k=50) 
    sparse_hits = bm25_retriever.invoke(query)
    
    
    unique_docs = {d.page_content: d for d in (dense_hits + sparse_hits)}.values()
    candidates = list(unique_docs)

    if not candidates:
        return []

    
    pairs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)
    
    scored_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    
    return [d for s, d in scored_candidates[:10]]

@app.post("/ask")
async def ask(question: Question):
    try:
        query = question.text
        loop = asyncio.get_running_loop()
        relevant_docs = await loop.run_in_executor(None, get_context, query)
        
        if not relevant_docs:
            return {"answer": "ინფორმაცია ვერ მოიძებნა.", "sources": []}

        context_parts = []
        sources = []
        for d in relevant_docs:
            src = d.metadata.get("case_number") or "Unknown"
            sources.append(src)
            context_parts.append(f"[წყარო: {src}]\n{d.page_content}")
        
        full_context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
        შენ ხარ RS.ge-ს საგადასახადო ექსპერტი . უპასუხე კითხვას მოცემული კონტექსტის საფუძველზე.
        კონტექსტი შეიცავს ოფიციალურ გადაწყვეტილებებს.
        
        წესები:
        1. თუ კითხვა ეხება კონკრეტულ ნომერს, გამოიყენე მხოლოდ ამ ნომრის მქონე წყარო.
        2. იყავი მაქსიმალურად ზუსტი ციფრებში და თარიღებში.
        3. თუ კონტექსტიდან გამომდინარე საჩივარი არ დაკმაყოფილდა, მკაფიოდ ახსენი მიზეზი.

        კონტექსტი:
        {full_context}

        კითხვა: {query}
        """

        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))

        return {
            "answer": response.text,
            "sources": list(set(sources))
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)