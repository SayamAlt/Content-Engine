## Content Engine Documentation

### Overview

The Content Engine is designed to analyze and compare multiple PDF documents using Retrieval Augmented Generation (RAG) techniques. It integrates a backend framework, vector store, embedding model, and local language model (LLM), along with a Streamlit frontend for user interaction.

### 1. Setup
#### a) Backend Framework
<ol type='a'>
  <li>LangChain
  A powerful toolkit for building LLM applications with a focus on retrieval-augmented generation.
  Installation instructions: pip install langchain</li>
  <li>Frontend Framework
Streamlit
An open-source app framework for creating interactive web applications.
Installation instructions: pip install streamlit</li>
  <li>Vector Store
ChromaDB
Chosen for its efficient management and querying of embeddings.
Setup instructions:
pip install chromadb
</li>
  <li>Embedding Model
Sentence Transformer
Local embedding model to generate embeddings from PDF content.
Installation:
pip install sentence-transformers</li>
  <li>Local Language Model (LLM)
Hugging Face Transformers
Integration of a local instance for processing and generating insights.
Installation:
  pip install transformers</li>
</ol>

### 2. Initialization

#### Data Preparation
Download and preprocess the three provided PDF documents (Alphabet Inc., Tesla Inc., Uber Technologies Inc.).

#### Parsing Documents
Use PyMuPDF or PyPDF2 to extract text and structure from PDFs.

#### Generating Vectors
Utilize Sentence Transformer to create embeddings for document content.

#### Storing in Vector Store
Implement functions to persist embeddings into ChromaDB vector store.

### 3. Development
#### Configuring Query Engine
Define retrieval tasks based on document embeddings using ChromaDB.

#### Integrating LLM
Set up a local instance of a Large Language Model (LLM) for contextual insights.

#### Developing Chatbot Interface
Use Streamlit to create a user-friendly interface for querying and displaying comparative insights from documents.

### 3. Usage
<ul>
  <li>Clone the repository:

git clone https://github.com/yourusername/content-engine.git
cd content-engine
</li>
 <li>Install dependencies:
pip install -r requirements.txt</li>
<li>Run the Streamlit app:
streamlit run content_engine.py</li>
</ul>
