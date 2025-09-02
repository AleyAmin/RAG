# 🏥 Ezz Medical RAG System GUI

If you encounter any error please delete the chroma and images(optional) folders then 
1) run the rag_gui.py file
2) go to the database tab
3) rebuild database
This should fix the problem

A comprehensive Retrieval-Augmented Generation (RAG) system designed for medical documents with an intuitive graphical user interface.

## 🌟 Features

### 💬 Intelligent Q&A Interface
- **Natural Language Queries**: Ask questions in plain English about your medical documents
- **Context-Aware Responses**: Get accurate answers based on your document content
- **Chat History**: Keep track of all your conversations with export functionality
- **Sample Questions**: Pre-built example questions to get you started quickly
- **Adjustable Results**: Control how many document chunks to consider for each query

### 🗄️ Advanced Document Management
- **Enhanced PDF Processing**: Automatically detects and processes tables, images, and structured content
- **Image Extraction**: Automatically extracts and saves images from PDFs to the `images` folder
- **Smart Text Chunking**: Intelligently splits documents while preserving context
- **Vector Database**: Uses Chroma DB with Google's Gemini embeddings for accurate similarity search
- **Batch Processing**: Add multiple documents at once

### 📊 Database Management
- **Real-time Status**: Monitor database health and document count
- **Rebuild Functionality**: Refresh the entire database when needed
- **Clear Database**: Remove all data when starting fresh
- **Process Logging**: Track all database operations with detailed logs
- **Image Statistics**: Track extracted images and their locations

### ⚙️ Configuration & Settings
- **API Key Management**: Easy setup for Google Gemini API
- **Path Configuration**: View and manage file locations
- **About Information**: System details and feature overview

## � Installation from ZIP File

### Prerequisites
- **Python 3.8 or higher** installed on your system
- **Google Gemini API key** (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))
- **Windows, macOS, or Linux**

### Step-by-Step Installation

1. **Extract the ZIP file**:
   - Download and extract the `Ezz-Medical-RAG-System.zip` file to your desired location

2. **Open Command Prompt/Terminal**:
   - **Windows**: Press `Win + R`, type `cmd`, press Enter
   - **macOS/Linux**: Open Terminal application

3. **Navigate to the project folder**:
   ```bash
   cd "path\to\extracted\Ezz-Medical-RAG-System"
   ```
   Example:
   ```bash
   cd "C:\Users\YourName\Documents\Ezz-Medical-RAG-System"
   ```

4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   If you get permission errors, try:
   ```bash
   pip install --user -r requirements.txt
   ```

5. **Launch the application**:
   **Direct launch**:
   ```bash
   python rag_gui.py
   ```

## 🚀 Quick Start Guide

### 1. First Time Setup

1. **Launch the application** using the methods above
2. **Add your documents -- optional**:
   - Go to the "🗄️ Database" tab
   - Click "📁 Add Documents"
   - Select your PDF files (medical papers, research documents, etc.)
3. **Build the database**:
   - Click "🔄 Rebuild Database"
   - Wait for processing to complete (images will be automatically extracted)
   - Check the process logs for detailed information

### 2. Asking Questions

1. **Go to the "💬 Ask Questions" tab**
2. **Try the sample questions**:
   - Click any of the 8 pre-written sample question buttons
   - Great for testing and learning what the system can do
3. **Or type your own question** in the text field
4. **Press Enter** or click "Ask"
5. **View the AI response** in the chat history with source citations
6. **Export your conversation** if needed (JSON or text format)

### 3. Sample Questions Feature
The GUI includes 8 carefully crafted sample questions:
- "What is the accuracy of BERT embeddings for redundancy detection?"
- "How does the system detect redundancy in software requirements?"
- "What are the main evaluation metrics used in this study?"
- "Summarize the methodology used for requirement analysis"
- "What are the advantages of using BERT over traditional methods?"
- "How many documents were used in the evaluation dataset?"
- "What is the precision and recall of the proposed system?"
- "Compare the performance with other state-of-the-art methods"

### 4. Managing Documents and Images

- **Add new documents**: Use "📁 Add Documents" to add more PDF files
- **Rebuild database**: Use "🔄 Rebuild Database" after adding new documents
- **Clear database**: Use "🗑️ Clear Database" to start fresh
- **View extracted images**: Check the `images` folder for automatically extracted images
- **Monitor status**: Check the database information panel for stats including image extraction counts
