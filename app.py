from pydoc import doc

from flask import Flask, request, jsonify, render_template
import os

print("🚀 Help desk....")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# ----------------------------
# Flask App
# ----------------------------

app = Flask(__name__)


# ----------------------------
# Load Multiple PDFs
# ----------------------------

documents = []
pdf_folder = "uploads"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)


# ----------------------------
# Text Splitter
# ----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)


# ----------------------------
# Embeddings
# ----------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------
# Vector Database
# ----------------------------

vectorstore = Chroma.from_documents(
    docs,
    embedding
)


# ----------------------------
# Retriever
# ----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ----------------------------
# Load LLM
# ----------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=150
)

llm = HuggingFacePipeline(
    pipeline=generator
)


# ----------------------------
# Create QA System
# ----------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

#RestAPI
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data=request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].lower().strip()
    
    #greetings
    if question in ["hi", "hello", "hey"]:
        return jsonify({"answer": "Hello! Welcome to GENAI Help Desk. How can I assist you today?"})
    #thank You
    if "thank" in question:
        return jsonify({"answer": "You're welcome! If you have any more questions, feel free to ask."})
    #bye
    if "bye" in question :
        return jsonify({"answer": "Goodbye! Have a great day!"})
    #AI Query 
    try:
        result=qa.invoke({
            "query": question
            })
        
        answer=result['result']

        doc=result["source_documents"][0]
        source = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", 0)
        return jsonify({"answer": answer, "source": source, "page": page})
    except Exception as e:
        return jsonify({"answer": "I'm sorry, I couldn't find an answer to your question"})
#run server
if __name__ == "__main__":
    app.run(debug=True)

# http://127.0.0.1:5000   that is the link to run the app in local server