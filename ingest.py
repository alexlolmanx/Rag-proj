import os
import re
import json
import shutil
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import torch

DATA_PATH = "data"
DB_PATH = "./chroma_db"

def normalize_text(s: str) -> str:
    if not s: return ""
    return s.replace("\xa0", " ").replace("\u200b", "").strip()


ORDER_RE = re.compile(r"N\s*(\d{4,6})")
CASE_RE = re.compile(r"(\d{1,6}/\d{1,3}/\d{4})")

def run_ingest():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"საქაღალდე '{DATA_PATH}' ცარიელია.")
        return

    if os.path.exists(DB_PATH):
        print("ძველი ბაზის წაშლა...")
        shutil.rmtree(DB_PATH)

    print("ინიციალიზაცია: e5-small (სწრაფი და მსუბუქი მოდელი)")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"ინდექსაცია ეშვება: {device}-ზე")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': device}
    )

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
    
    vector_db = None
    current_batch = []
    batch_size = 150  

    print(f"ვიწყებ {len(all_files)} ფაილის დამუშავებას ნაწილ-ნაწილ...")

    for i, fname in enumerate(tqdm(all_files, desc="ინდექსაცია")):
        path = os.path.join(DATA_PATH, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            text = normalize_text(data.get("document", {}).get("content", ""))
            if not text: continue

            
            doc_type = "გადაწყვეტილება" if "გადაწყვეტილება" in text[:300] else "ბრძანება"
            order_match = ORDER_RE.search(text[:300])
            case_match = CASE_RE.search(text[:300])
            
            num_val = order_match.group(1) if order_match else (case_match.group(1) if case_match else None)

            
            if num_val:
                title = f"დოკუმენტის #:{num_val}"
            else:
                
                title = f"დოკუმენტი {fname[:8]}"

            current_batch.append(Document(page_content=text, metadata={"source": title, "case_number": num_val}))


           
            if len(current_batch) >= batch_size:
                chunks = text_splitter.split_documents(current_batch)
                if vector_db is None:
                    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
                else:
                    vector_db.add_documents(chunks)
                
                current_batch = [] 
                
        except Exception as e:
            continue

    
    if current_batch:
        chunks = text_splitter.split_documents(current_batch)
        if vector_db:
            vector_db.add_documents(chunks)
        else:
            vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

    print("ინგესტი წარმატებით დასრულდა")
if __name__ == "__main__":
    run_ingest()


    