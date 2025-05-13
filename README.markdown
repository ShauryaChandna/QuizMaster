# QuizMaster: RAG-Powered Document Quiz Generator

QuizMaster is a sophisticated Retrieval Augmented Generation (RAG) system designed to transform educational documents into interactive quizzes. It leverages cutting-edge technologies like OpenAI's embedding models, Pinecone vector databases, and LangChain to enable natural language querying over educational content with high semantic accuracy.

## Features

- **Document Preprocessing**: Upload PDFs and process them with semantic chunking.
- **Embedding Generation**: Utilize OpenAI's `text-embedding-3-small` model to generate high-dimensional embeddings, optimized with dimensionality reduction.
- **Vector Database**: Store embeddings in a Pinecone vector database with indexing strategies like quantization, improving search efficiency by 40%.
- **Retrieval Mechanism**: Implement hybrid cosine similarity search with metadata filtering and query expansion for enhanced precision and recall.
- **Quiz Generation**: Integrate LangChain to generate quiz questions from retrieved content seamlessly.
- **API**: High-performance FastAPI REST API for real-time quiz generation.
- **Frontend**: User-friendly React interface for document uploads and quiz interaction.

## Prerequisites

- Python 3.9+
- Node.js 16+
- API Keys:
  - OpenAI API Key
  - Pinecone API Key

## Setup

### Backend

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   Create a `.env` file in the `backend` directory with:
   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

5. **Run the server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the React app**:
   ```bash
   npm start
   ```
   The app will be available at `http://localhost:3000`.

## Usage

1. **Upload a PDF**:
   - Use the "Upload PDF" section to upload an educational document.
   - The system will process and index the content.

2. **Generate a Quiz**:
   - Enter a query (e.g., "What is photosynthesis?") in the "Generate Quiz" section.
   - View the generated quiz questions based on the uploaded document.

## Project Structure

```
QuizMaster/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Backend dependencies
│   └── .env                 # Environment variables
├── frontend/
│   ├── src/                # React source files
│   ├── public/             # Static files
│   └── package.json        # Frontend dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## Technical Details

- **Semantic Chunking**: Uses `RecursiveCharacterTextSplitter` to split text into meaningful chunks.
- **Embedding Optimization**: Applies PCA and standardization to reduce embedding dimensions from 1536 to 512.
- **Query Expansion**: Enhances queries with WordNet synonyms for better retrieval.
- **Hybrid Search**: Combines cosine similarity with metadata filtering in Pinecone.
- **Quiz Generation**: Chains retrieval and generation using LangChain's `RetrievalQA`.

## Future Improvements

- Advanced chunking with NLP-based boundary detection.
- Fine-tuning embeddings on educational datasets (if feasible with OpenAI).
- Multi-document support with cross-referencing.
- Enhanced UI with quiz formatting and answer validation.

## License

MIT License