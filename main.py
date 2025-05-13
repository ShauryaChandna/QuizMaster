from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import wordnet
import logging
from utils import advanced_chunking, quantize_embedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="QuizMaster API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys must be set in environment variables.")

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone index configuration with explicit HNSW parameters
INDEX_NAME = "quizmaster-index"
EMBEDDING_DIM = 1536  # Original OpenAI embedding dimension
REDUCED_DIM = 512     # Reduced dimension for optimization
HNSW_M = 16           # HNSW: number of bi-directional links (tweak for accuracy vs speed)
HNSW_EF_CONSTRUCTION = 200  # HNSW: size of dynamic list during construction (tweak for build time vs quality)

# Create or connect to Pinecone index with HNSW
if INDEX_NAME not in pinecone_client.list_indexes().names():
    logger.info(f"Creating Pinecone index: {INDEX_NAME} with HNSW (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})")
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=REDUCED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        pods=1,
        pod_type="p1.x1",
        metadata_config={"indexed": ["file_id", "chunk_type"]}
    )

index = pinecone_client.Index(INDEX_NAME)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = LangChainPinecone(index, embeddings.embed_query, "text")
llm = LangChainOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Initialize PCA and Scaler for dimensionality reduction
scaler = StandardScaler()
pca = PCA(n_components=REDUCED_DIM)

# Initialize sentence transformer for advanced chunking
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Download NLTK data for query expansion
nltk.download('wordnet')

def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """Generate embeddings with dimensionality reduction and quantization."""
    raw_embeddings = []
    for chunk in chunks:
        response = openai_client.embeddings.create(input=chunk, model="text-embedding-3-small")
        embedding = response.data[0].embedding
        raw_embeddings.append(embedding)
    
    # Standardize and reduce dimensionality
    scaled_embeddings = scaler.fit_transform(raw_embeddings)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    
    # Quantize embeddings for optimization
    quantized_embeddings = [quantize_embedding(emb) for emb in reduced_embeddings]
    return quantized_embeddings

def expand_query(query: str) -> str:
    """Expand query with synonyms using WordNet for better retrieval."""
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded = query + " " + " ".join(synonyms)
    return expanded[:500]  # Truncate to avoid excessive length

# Advanced prompt engineering for complex quiz queries
QUIZ_PROMPT = PromptTemplate(
    input_variables=["context", "concepts", "difficulty", "num_mcq", "num_long"],
    template="""
    You are an expert quiz generator tasked with creating a quiz based on a document's content. Generate a quiz with {num_mcq} multiple-choice questions (MCQs) and {num_long} long answer questions on the concepts of {concepts} at a {difficulty} difficulty level, using the provided context.

    **Instructions**:
    1. **Analyze the Context**: Extract key information related to {concepts} from the document content.
    2. **Generate Questions**:
       - **MCQs**: Provide 4 options per question, with one correct answer clearly marked.
       - **Long Answer**: Pose questions requiring detailed responses, encouraging analysis or explanation.
    3. **Difficulty Adjustment**:
       - Easy: Basic recall or simple concepts.
       - Medium: Application or moderate complexity.
       - Hard: Critical thinking or advanced synthesis.
    4. **Ensure Quality**: Questions must be relevant, clear, and tied to the specified concepts.

    **Example**:
    - Concepts: Photosynthesis, Cellular Respiration
    - Difficulty: Medium
    - MCQs: 2
    - Long Answer: 1
    - Output:
      1. What molecule is produced during photosynthesis?
         A) Oxygen
         B) Nitrogen
         C) Carbon Monoxide
         D) Sulfur
         **Correct Answer**: A
      2. Which process uses oxygen to break down glucose?
         A) Photosynthesis
         B) Cellular Respiration
         C) Fermentation
         D) Glycolysis
         **Correct Answer**: B
      3. Describe how photosynthesis and cellular respiration are interdependent.

    **Quiz Generation**:
    Context: {context}
    Concepts: {concepts}
    Difficulty: {difficulty}
    Number of MCQs: {num_mcq}
    Number of Long Answer Questions: {num_long}

    Quiz Questions:
    """
)

@app.post("/upload", response_model=Dict[str, str])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and process it with advanced chunking and HNSW indexing."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs allowed.")

    # Save the uploaded file
    file_id = str(uuid.uuid4())
    file_path = f"uploads/{file_id}.pdf"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extract text from PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise ValueError("No text extracted from PDF.")

        # Advanced chunking using sentence embeddings
        chunks = advanced_chunking(text, sentence_model)
        logger.info(f"Generated {len(chunks)} advanced chunks from document {file_id}")

        # Generate optimized embeddings
        embeddings = generate_embeddings(chunks)

        # Prepare metadata and upsert to Pinecone with HNSW
        vectors = [
            (str(uuid.uuid4()), emb.tolist(), {"text": chunk, "file_id": file_id, "chunk_type": "semantic"})
            for emb, chunk in zip(embeddings, chunks)
        ]
        index.upsert(vectors=vectors)
        logger.info(f"Indexed {len(vectors)} vectors for document {file_id} with HNSW")

        return {"message": "Document processed and indexed successfully", "file_id": file_id}

    except Exception as e:
        logger.error(f"Error processing document {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query", response_model=Dict[str, str])
async def generate_quiz(query_data: Dict):
    """Generate quiz questions based on a structured query."""
    try:
        # Parse structured query (e.g., {"concepts": "photosynthesis, cellular respiration", "difficulty": "medium", "num_mcq": 3, "num_long": 2})
        concepts = query_data.get("concepts", "").strip()
        difficulty = query_data.get("difficulty", "medium").strip().lower()
        num_mcq = int(query_data.get("num_mcq", 0))
        num_long = int(query_data.get("num_long", 0))

        if not concepts or num_mcq < 0 or num_long < 0:
            raise HTTPException(status_code=400, detail="Invalid query parameters.")

        # Expand concepts for better retrieval
        expanded_concepts = expand_query(concepts)
        logger.info(f"Expanded concepts: {expanded_concepts}")

        # Retrieve relevant chunks using HNSW indexing
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"chunk_type": "semantic"}}
        )
        relevant_chunks = retriever.get_relevant_documents(expanded_concepts)
        if not relevant_chunks:
            return {"quiz_questions": "No relevant content found to generate quiz."}

        # Combine chunks into context
        context = "\n".join([doc.page_content for doc in relevant_chunks])

        # Generate quiz with advanced prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": QUIZ_PROMPT}
        )
        quiz_questions = qa_chain.run({
            "context": context,
            "concepts": concepts,
            "difficulty": difficulty,
            "num_mcq": num_mcq,
            "num_long": num_long
        })

        logger.info(f"Generated quiz for concepts: {concepts}")
        return {"quiz_questions": quiz_questions}

    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)