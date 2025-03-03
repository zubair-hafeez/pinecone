from dotenv import load_dotenv
from flask import Flask, render_template, request
import json
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import requests
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

pinecone_index_name = "question-answering-chatbot"
DATA_DIR = "tmp"
DATA_FILE = f"{DATA_DIR}/quora_duplicate_questions.tsv"
DATA_URL = "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def delete_existing_pinecone_index():
    if pinecone_index_name in pc.list_indexes().names():
        pc.delete_index(pinecone_index_name)

def create_pinecone_index():
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=300,  # Corrected dimension for GloVe 300d embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(pinecone_index_name)

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        r = requests.get(DATA_URL)
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)

def read_tsv_file():
    df = pd.read_csv(DATA_FILE, sep="\t", usecols=["qid1", "question1"], index_col=False)
    df.drop_duplicates(inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def create_and_apply_model():
    model = SentenceTransformer("average_word_embeddings_glove.6B.300d")

    df["question_vector"] = df["question1"].apply(lambda x: model.encode(str(x)).tolist())

    vectors = list(zip(df["qid1"].astype(str), df["question_vector"]))
    pinecone_index.upsert(vectors)

    return model

def query_pinecone(search_term):
    query_question = str(search_term)
    query_vector = model.encode(query_question).tolist()

    query_results = pinecone_index.query(vector=query_vector, top_k=5, include_values=False)
    res = query_results["matches"]

    results_list = []
    for match in res:
        qid = match["id"]
        score = match["score"]
        matched_question = df[df.qid1.astype(str) == qid]["question1"].values

        if len(matched_question) > 0:
            results_list.append({"id": qid, "question": matched_question[0], "score": score})

    return json.dumps(results_list)

delete_existing_pinecone_index()
pinecone_index = create_pinecone_index()
download_data()
df = read_tsv_file()
model = create_and_apply_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        return query_pinecone(request.form.get("question", ""))
    if request.method == "GET":
        return query_pinecone(request.args.get("question", ""))
    return "Only GET and POST methods are allowed for this endpoint"

if __name__ == "__main__":
    app.run(debug=True)
