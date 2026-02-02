# Django AI Chat with RAG

A production-ready Django application that implements Retrieval-Augmented Generation (RAG) using FAISS as the vector database. The application supports multiple embedding providers (OpenAI, Gemini, HuggingFace) and provides a modern chat interface for querying indexed documents.

## Features

- **Multiple Embedding Providers**: OpenAI, Google Gemini, and HuggingFace (local)
- **Multiple LLM Providers**: OpenAI, Google Gemini, and Anthropic
- **FAISS Vector Store**: Efficient similarity search with disk persistence
- **Document Support**: PDF, TXT, MD, and DOCX files
- **Configurable Chunking**: Character, sentence, and paragraph-based chunking strategies
- **Modern Chat UI**: Responsive interface with Bootstrap 5
- **Django Admin**: Full admin interface for configuration and chat history
- **Management Commands**: Easy document indexing via CLI

## Architecture

```
djgent-chat/
├── config/              # Django project configuration
├── chat/                # Main chat application
│   ├── services/         # Business logic services
│   │   ├── base.py                    # Abstract base classes
│   │   ├── embedding_service.py        # Embedding providers
│   │   ├── vector_store_service.py     # FAISS vector store
│   │   ├── document_service.py        # Document loaders
│   │   ├── chunking_service.py       # Text chunking
│   │   └── rag_service.py           # RAG orchestration
│   ├── utils/             # Utility functions
│   ├── management/        # Django management commands
│   ├── templates/         # HTML templates
│   └── static/           # CSS and JavaScript
├── data/documents/       # Document storage
├── faiss_indexes/       # FAISS index storage
└── .env.example         # Environment variables template
```

## Installation

### Prerequisites

- Python 3.12+
- pip or uv package manager

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd djgent-chat
   ```

2. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (if configured)
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (optional, for admin access)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Prepare documents**
   ```bash
   mkdir -p data/documents/default
   # Place your PDF, TXT, MD, or DOCX files in data/documents/default/
   ```

7. **Index documents**
   ```bash
   python manage.py index_documents --folder default
   ```

8. **Run the development server**
   ```bash
   python manage.py runserver
   ```

9. **Access the application**
   - Chat Interface: http://localhost:8000/chat/
   - Settings: http://localhost:8000/chat/settings/
   - Admin: http://localhost:8000/admin/

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider (openai, gemini, anthropic)
LLM_PROVIDER=openai

# API Keys
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Embedding Provider (openai, gemini, huggingface)
EMBEDDING_PROVIDER=openai

# HuggingFace Model (for local embeddings)
HUGGINGFACE_MODEL_NAME=all-MiniLM-L6-v2

# RAG Configuration
RAG_FAISS_INDEX_BASE_PATH=faiss_indexes
RAG_DOCUMENTS_BASE_PATH=data/documents

# Chunking Configuration
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
DEFAULT_TOP_K=5
```

### Django Admin Configuration

You can also configure RAG settings via the Django Admin interface:

1. Navigate to http://localhost:8000/admin/
2. Go to "RAG Configurations"
3. Create or edit a configuration with:
   - **Documents Folder Name**: Name of the folder containing documents
   - **Embedding Provider**: Provider for generating embeddings
   - **LLM Provider**: Provider for generating responses
   - **Chunk Size**: Target size of each chunk (characters)
   - **Chunk Overlap**: Overlap between chunks (characters)
   - **Top K**: Number of relevant chunks to retrieve

## Usage

### Indexing Documents

Index documents using the management command:

```bash
# Basic indexing
python manage.py index_documents --folder default

# Force re-indexing (overwrite existing index)
python manage.py index_documents --folder default --reindex

# Custom chunking parameters
python manage.py index_documents --folder default --chunk-size 1500 --chunk-overlap 300

# Use specific embedding provider
python manage.py index_documents --folder default --embedding-provider huggingface

# Use sentence-based chunking
python manage.py index_documents --folder default --chunking-strategy sentence

# Index specific file types only
python manage.py index_documents --folder default --extensions pdf,txt

# Use absolute path for documents
python manage.py index_documents --folder-path /path/to/documents
```

### Chat Interface

1. Navigate to http://localhost:8000/chat/
2. Type your question in the input field
3. Press Enter or click Send
4. The AI will retrieve relevant chunks and generate a response
5. Sources are displayed below each response

### Starting a New Conversation

Click the "+" button in the chat header to start a new conversation session.

## API Endpoints

### Chat API

**POST** `/chat/api/chat/`

Send a message and get an AI response.

**Request Body:**
```json
{
  "message": "What is the main topic of the documents?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "file_name": "document.pdf",
      "source_path": "/path/to/document.pdf",
      "chunk_count": 3
    }
  ],
  "context": [...],
  "session_id": "session-id"
}
```

### Conversation API

**GET** `/chat/api/conversation/<session_id>/`

Retrieve conversation history for a session.

**Response:**
```json
{
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "Question",
      "sources": [],
      "created_at": "2024-01-01T12:00:00Z"
    }
  ],
  "session_id": "session-id"
}
```

### Index Status API

**GET** `/chat/api/index-status/`

Get current index status.

**Response:**
```json
{
  "is_ready": true,
  "vector_count": 1234,
  "dimension": 1536,
  "index_path": "/path/to/index"
}
```

### New Session API

**GET** `/chat/api/new-session/`

Create a new chat session.

**Response:**
```json
{
  "session_id": "new-session-uuid"
}
```

## Development

### Running Tests

```bash
python manage.py test
```

### Creating Custom Embedding Providers

1. Create a new class in `chat/services/embedding_service.py` that extends `EmbeddingProvider`
2. Implement the required methods: `get_default_model()`, `get_dimension()`, `embed_text()`, `embed_batch()`
3. Add the provider to the `get_embedding_provider()` factory function
4. Update the model choices in `chat/models.py`

### Creating Custom Document Loaders

1. Create a new class in `chat/services/document_service.py` that extends `DocumentLoader`
2. Implement the required methods: `load_document()`, `get_supported_extensions()`, `can_load()`
3. Add the loader to the `DocumentLoader.LOADERS` list

## Troubleshooting

### No documents indexed error

If you see "No documents indexed" error:
1. Ensure you've run `python manage.py index_documents`
2. Check that documents exist in `data/documents/[folder_name]/`
3. Verify API keys are configured in `.env`

### FAISS import error

If you get an import error for FAISS:
```bash
pip install faiss-cpu
# or for GPU support
pip install faiss-gpu
```

### Embedding provider errors

- **OpenAI**: Ensure `OPENAI_API_KEY` is set in `.env`
- **Gemini**: Ensure `GEMINI_API_KEY` is set in `.env`
- **HuggingFace**: First run will download the model automatically

### Document loading errors

- **PDF**: Ensure PDF files are not password-protected
- **DOCX**: Ensure files are valid DOCX format
- **Encoding issues**: TXT files should be UTF-8 encoded

## Production Deployment

### Security Considerations

1. Set `DEBUG = False` in production
2. Use a strong `SECRET_KEY`
3. Configure `ALLOWED_HOSTS` with your domain
4. Use environment variables for all API keys
5. Enable HTTPS
6. Configure proper file permissions for document storage

### Static Files

```bash
python manage.py collectstatic
```

### Database

For production, consider using PostgreSQL instead of SQLite:

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'djgent_chat',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

## License

This project is provided as-is for educational and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
