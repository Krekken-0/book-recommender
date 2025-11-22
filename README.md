# Book Recommender System

An intelligent book recommendation system that uses semantic search, vector embeddings, and emotion analysis to provide personalized book recommendations. The system enables users to search for books using natural language queries and filter results by category and emotional tone through an intuitive Gradio web interface.

## Prerequisites

Before installing and running this project, ensure you have the following installed:

- **Python** 3.8 or higher
- **pip** (Python package installer)

## Installation Guide

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd book-recommender
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install all required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary Python packages including:
   - `pandas` and `numpy` for data manipulation
   - `chromadb` for vector database operations
   - `langchain` and related packages for embedding and document processing
   - `gradio` for the web interface
   - `transformers` and `sentence-transformers` for NLP models
   - Additional dependencies for emotion analysis and vector search

4. **Ensure data files are available**:
   - The project requires `books_with_emotion.csv` (included in the repository)
   - The ChromaDB vector database is located in `chroma_db_new/` directory
   - A fallback cover image `cover-not-found.jpg` should be present

## Running the Project

To start the book recommendation web interface, run:

```bash
python gradio-dashboard.py
```

The application will launch a Gradio web interface. You should see output indicating the local URL (typically `http://127.0.0.1:7860`). Open this URL in your web browser to access the recommendation system.

### Usage

Once the interface is running:

1. Enter a natural language description of the book you're looking for (e.g., "A story about forgiveness" or "A book to teach children about nature")
2. Select a category filter (optional): All, Children's Fiction, Children's Nonfiction, Fiction, or Nonfiction
3. Select an emotional tone filter (optional): All, Happy, Surprising, Angry, Suspenseful, or Sad
4. Click "Find recommendations" to see personalized book suggestions with covers and descriptions

## Project Structure

```
book-recommender/
├── gradio-dashboard.py          # Main application entry point (Gradio interface)
├── books_with_emotion.csv       # Dataset with book metadata and emotion scores
├── books_cleaned.csv            # Cleaned book dataset
├── books_with_categories.csv    # Book dataset with categorized information
├── chroma_db_new/               # ChromaDB vector database for semantic search
├── cover-not-found.jpg          # Fallback image for books without cover art
├── tagged_descriptions.txt      # Preprocessed book descriptions for vectorization
├── data-exploration.ipynb       # Jupyter notebook for data exploration
├── sentiment-analysis.ipynb     # Notebook for emotion analysis pipeline
├── text-classification.ipynb    # Notebook for category classification
├── vector-search.ipynb          # Notebook for vector database setup and testing
├── requirements.txt             # Python package dependencies
└── README.md                    # This file
```

### Key Components

- **`gradio-dashboard.py`**: The main application that provides the web interface for book recommendations using semantic search
- **Vector Database (`chroma_db_new/`)**: Stores embedded book descriptions for fast similarity search
- **Notebooks**: Jupyter notebooks for data preprocessing, emotion analysis, and system development
- **Data Files**: CSV files containing book metadata, categories, and emotion scores

## How It Works

1. **Semantic Search**: Uses HuggingFace's `all-MiniLM-L6-v2` model to generate embeddings of book descriptions
2. **Vector Database**: ChromaDB stores and retrieves similar books based on user queries
3. **Filtering**: Results are filtered by category and sorted by emotional tone scores
4. **Recommendations**: Returns up to 16 relevant book recommendations with covers and descriptions

