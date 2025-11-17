import chromadb
from chromadb.config import Settings
import streamlit as st
from openai import OpenAI

from utils import read_pdf, get_embeddings, chunk_text, hash_id

st.title("ChatBot Demo")

# sidebar settings
st.sidebar.header("API Keys")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="Enter a API Key starting with sk-",
)

# init clients
openai_client = OpenAI(api_key=api_key)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("paper_rag")

# initialize session state
if "message" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = ""

if not api_key:
    st.warning("Enter a OpenAI API Key to left sidebar")
    st.stop()


file_uploader = st.sidebar.file_uploader(
    "PDF Uploader", type=["pdf"], help="Upload a PDF file you want to ask about"
)
if file_uploader is not None and st.session_state.pdf_file != file_uploader.name:
    try:
        text = read_pdf(file_uploader)
        chunks = chunk_text(text)
        ids = [hash_id(file_uploader.name, c) for c in chunks]
        hashmap = {}
        for id, chunk in zip(ids, chunks):
            hashmap[id] = chunk

        # filter already existing chunk
        exist_ids = collection.get(ids=ids)["ids"]
        ids = [id for id in ids if id not in exist_ids]
        new_chunks = []
        for id in ids:
            new_chunks.append(hashmap[id])

        # get embeds
        embeds = get_embeddings(new_chunks, openai_client)

        # remove all items in the collection
        # collections = chroma_client.list_collections()
        # collection_names = [c.name for c in collections]
        # if "paper_rag" in collection_names:
        #     chroma_client.delete_collection("paper_rag")
        # collection = chroma_client.get_or_create_collection("paper_rag")

        # insert new items
        collection.add(documents=chunks, embeddings=embeds, ids=ids)

        st.session_state.pdf_file = file_uploader.name
        st.success("Successfully read a PDF")
    except Exception as e:
        st.error(f"Error occurred while reading the PDF: {e}")

# print previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user inputs
user_input = st.chat_input("Type here...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    chat_history = []
    chat_history.append({"role": "system", "content": ""})
    for msg in st.session_state.messages:
        chat_history.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # make a query embed from a user input
            query_embed = (
                openai_client.embeddings.create(
                    model="text-embedding-3-small", input=user_input
                )
                .data[0]
                .embedding
            )

            # get top 5 reference docs in ChromaDB
            result = collection.query(query_embeddings=[query_embed], n_results=5)
            rag = "\n\n---\n\n".join(result["documents"][0])

            user_prompt = f"""
[Chat History]
{chat_history}

[Reference]
{rag}

[Question]
{user_input}

By referencing docs in [Reference] to answer user's question
"""

            response = openai_client.responses.create(
                model="gpt-4.1-mini", input=user_prompt
            )
            assistant_text = response.output[0].content[0].text
            st.markdown(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
