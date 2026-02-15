import streamlit as st
import os
import re
import torch
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


st.set_page_config(page_title="RS.ge AI Assistant", page_icon="⚖️")
st.title("⚖️ RS.ge საგადასახადო მრჩეველი (RAG)")


api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("გთხოვთ, დაამატოთ GOOGLE_API_KEY Space-ის Settings-ში!")


@st.cache_resource
def init_models():
    device = "cpu" 
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': device}
    )
    
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    return vector_db, reranker

vector_db, reranker = init_models()

# ჩატის ისტორია
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_context(query):
    
    case_num_match = re.search(r"(\d{4,6})", query)
    if case_num_match:
        case_id = case_num_match.group(1)
        results = vector_db.get(where={"case_number": case_id})
        if results and results['documents']:
            return "\n\n".join(results['documents']), [case_id]

  
    docs = vector_db.similarity_search(query, k=15)
    
  
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    top_docs = [d for s, d in scored_docs[:5]]
    context = "\n\n---\n\n".join([f"[Case: {d.metadata.get('case_number')}]\n{d.page_content}" for d in top_docs])
    sources = [d.metadata.get('case_number') for d in top_docs]
    return context, sources


if prompt := st.chat_input("დასვით კითხვა საგადასახადო დავაზე..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context, sources = get_context(prompt)
        
        gemini_prompt = f"კონტექსტი:\n{context}\n\nკითხვა: {prompt}\nუპასუხე ქართულად, იყავი ზუსტი."
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(gemini_prompt)
        
        full_response = f"{response.text}\n\n**წყაროები:** {', '.join(list(set(sources)))}"
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})