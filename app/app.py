import os
import tempfile
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
import uuid

load_dotenv()

st.info("🔒 Your API key will only be used temporarily during this session and not stored.")
user_openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

if not user_openai_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Set it as environment variable for OpenAI client
os.environ["OPENAI_API_KEY"] = user_openai_key

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


os.environ["QDRANT_HOST"] = st.secrets.get("QDRANT_HOST", os.getenv("QDRANT_HOST"))
os.environ["QDRANT_API_KEY"] = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))

# QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION = "2nd-use-of-vectors"



st.set_page_config(page_title=" PDF RAG Chatbot 📄", layout="wide")
st.title("PDF RAG Chatbot 📄")

# Making a Qdrant client for DB
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)
# Check if collection exists, create if not
def ensure_collection():
    collections = qdrant.get_collections().collections
    names = [col.name for col in collections]
    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

# Have extract text per page of pdf file
def extract_pages(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]

# Chunking
def chunk_pages(pages):
    chunks = []
    for page_num, content in enumerate(pages, start=1):
        if not content.strip():
            continue
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for chunk in splitter.split_text(content):
            chunks.append({
                "page": page_num,
                "text": chunk
            })
    return chunks

# Vector Embeddings and storing the [embedding] in qdrant
def embed_and_store(chunks, base_pdf_url=None):
    ensure_collection()
    vectors = []
    for chunk in chunks:
        full_text = chunk["text"]
        embedding = client.embeddings.create(
            input=[full_text],
            model="text-embedding-3-small"
        ).data[0].embedding

        metadata = {
            "text": full_text,
            "page": chunk["page"],
            "link": f"{base_pdf_url}#page={chunk['page']}" if base_pdf_url else ""
        }

        vectors.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=metadata
        ))

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=vectors)

# Vector Similarity Search [query] in DB
def search_similar(query):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    search = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=3,
        with_payload=True
    )
    return search

# To get answer with chatbot
def generate_answer(context, question):
    context_text = "\n\n".join([f"Page {r.payload['page']}:\n{r.payload['text']}" for r in context])
    SYSTEM_PROMPT = f"""    
        You are an helpful AI Assistant who answers user query based on the availabe context retrieved from a PDF file along with page_contents and page number.

        You should only answer the user based on the following context and navigate the user to open the right page number to know more.

        Context:
        {context_text}

        Question: {question}
        Answer:
        """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": SYSTEM_PROMPT}]
    )

    return response.choices[0].message.content

# Now have to apply Streamlit UI
uploaded_file = st.file_uploader("Upload PDF 📤", type=["pdf"])
base_pdf_url = st.text_input("🔗 (Optional) Paste public URL of the same PDF (to link to pages)")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("SUCCESS ✅ : PDF uploaded. Processing...")

    pages = extract_pages(tmp_path)
    chunks = chunk_pages(pages)
    embed_and_store(chunks, base_pdf_url or None)
    st.success("SUCCESS ✅ : Text embedded & stored in Qdrant.")

    st.subheader("🤗: Ask anything from or about this document/PDF")
    user_query = st.text_input("😨: Your question")

    if user_query:
        with st.spinner("🔍 Searching and generating answer..."):
            results = search_similar(user_query)
            answer = generate_answer(results, user_query)

            st.markdown("### Your Answer : ")
            st.write(answer)

            st.markdown("### 🔗 Source Pages")
            seen_pages = set()
            for r in results:
                page = r.payload["page"]
                if page in seen_pages:
                    continue
                seen_pages.add(page)

                link = r.payload.get("link", "")
                st.markdown(f"📄 Page {page}: [View Page]({link})" if link else f"📄 Page {page}")


