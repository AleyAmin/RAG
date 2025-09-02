import argparse
import os
import shutil
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import embedding_function
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import pdfplumber
from PIL import Image
import fitz  

load_dotenv()

genai.api_key = os.environ['GEMINI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "content"
IMAGES_PATH = "images"

# def load_db(file, chain_type, k):
#     # load documents
#     loader = PyPDFLoader(file)
#     documents = loader.load()
#     # split documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     docs = text_splitter.split_documents(documents)
#     # define embedding
#     embeddings = OpenAIEmbeddings()
#     # create vector database from data
#     db = DocArrayInMemorySearch.from_documents(docs, embeddings)
#     # define retriever
#     retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
#     # create a chatbot chain. Memory is managed externally.
#     qa = ConversationalRetrievalChain.from_llm(
#         llm=ChatOpenAI(model_name=llm_name, temperature=0), 
#         chain_type=chain_type, 
#         retriever=retriever, 
#         return_source_documents=True,
#         return_generated_question=True,
#     )
#     return qa 

def extract_images_from_pdf(pdf_path, filename_base):
    """Extract images from PDF and save them to the images folder"""
    images_saved = []
    
    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_PATH, exist_ok=True)
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get the XREF of the image
                    xref = img[0]
                    
                    # Extract the image
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Convert CMYK to RGB if necessary
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    
                    # Save the image
                    img_filename = f"{filename_base}_page{page_num + 1}_img{img_index + 1}.png"
                    img_path = os.path.join(IMAGES_PATH, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    
                    images_saved.append({
                        'filename': img_filename,
                        'page': page_num + 1,
                        'path': img_path,
                        'width': pix.width,
                        'height': pix.height
                    })
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
                    continue
        
        pdf_document.close()
        
    except ImportError:
        print("    📝 PyMuPDF not available, falling back to pdfplumber for image detection only")
        # Fallback to pdfplumber for image detection (without extraction)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                images = page.images
                if images:
                    for i, img in enumerate(images):
                        images_saved.append({
                            'filename': f"{filename_base}_page{page_num + 1}_img{i + 1}_metadata_only",
                            'page': page_num + 1,
                            'path': None,  # No actual file saved
                            'width': img.get('width', 'unknown'),
                            'height': img.get('height', 'unknown')
                        })
    
    except Exception as e:
        print(f"    ⚠️ Error extracting images: {e}")
    
    return images_saved

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    documents = []
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DATA_PATH, filename)
            filename_base = os.path.splitext(filename)[0]
            print(f"📑 Processing {filename} with enhanced extraction...")
            
            # Extract images first
            print(f"  🖼️ Extracting images...")
            extracted_images = extract_images_from_pdf(file_path, filename_base)
            if extracted_images:
                saved_images = [img for img in extracted_images if img.get('path')]
                detected_images = [img for img in extracted_images if not img.get('path')]
                
                if saved_images:
                    print(f"    ✅ Saved {len(saved_images)} image(s) to '{IMAGES_PATH}' folder")
                if detected_images:
                    print(f"    📋 Detected {len(detected_images)} image(s) (metadata only)")
            else:
                print(f"    📋 No images found")
            
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text with better formatting
                        text = page.extract_text() or ""
                        
                        # Enhanced image content with extraction info
                        image_content = ""
                        page_images = [img for img in extracted_images if img['page'] == page_num + 1]
                        if page_images:
                            print(f"  🖼️ Page {page_num + 1}: Found {len(page_images)} image(s)")
                            image_descriptions = []
                            for img in page_images:
                                if img.get('path'):
                                    desc = f"Image: {img['filename']} ({img['width']}x{img['height']} pixels) - Saved to: {img['path']}"
                                else:
                                    desc = f"Image detected: {img['width']}x{img['height']} pixels (metadata only)"
                                image_descriptions.append(desc)
                            
                            image_content = f"\\n\\n[IMAGES ON THIS PAGE]\\n" + "\\n".join(image_descriptions) + "\\n[/IMAGES]\\n\\n"
                        
                        # Try to extract tables (even if not perfectly structured)
                        tables = page.extract_tables()
                        table_content = ""
                        if tables:
                            print(f"  📊 Page {page_num + 1}: Found {len(tables)} table(s)")
                            for i, table in enumerate(tables):
                                table_content += f"\\n\\n[TABLE {i+1}]\\n"
                                for row in table:
                                    if row and any(cell for cell in row if cell):  # Skip empty rows
                                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                                        table_content += " | ".join(clean_row) + "\\n"
                                table_content += "[/TABLE]\\n\\n"
                        
                        # Look for table-like patterns in text (fallback)
                        table_keywords = ['accuracy', 'precision', 'recall', 'f1-score', 'results', 'evaluation', 'performance']
                        if any(keyword in text.lower() for keyword in table_keywords) and not tables:
                            # Mark potential table sections
                            lines = text.split('\\n')
                            for i, line in enumerate(lines):
                                if any(keyword in line.lower() for keyword in table_keywords):
                                    # Check surrounding lines for numeric data
                                    context_start = max(0, i-2)
                                    context_end = min(len(lines), i+3)
                                    context = lines[context_start:context_end]
                                    
                                    # Look for lines with numbers/percentages
                                    numeric_lines = [l for l in context if any(c.isdigit() for c in l) and '%' in l]
                                    if numeric_lines:
                                        table_content += f"\\n\\n[POTENTIAL_TABLE_SECTION]\\n"
                                        table_content += "\\n".join(numeric_lines)
                                        table_content += "\\n[/POTENTIAL_TABLE_SECTION]\\n\\n"
                                        break
                        
                        # Combine all content
                        full_content = text + image_content + table_content
                        
                        # Create document with enhanced metadata
                        doc = Document(
                            page_content=full_content,
                            metadata={
                                'source': file_path,
                                'page': page_num,
                                'total_pages': len(pdf.pages),
                                'processing_type': 'enhanced',
                                'images_found': len(page_images),
                                'images_extracted': len([img for img in page_images if img.get('path')]),
                                'tables_found': len(tables) if tables else 0,
                                'has_table_keywords': any(keyword in text.lower() for keyword in table_keywords)
                            }
                        )
                        documents.append(doc)
                
                print(f"  ✅ Processing completed: {len(pdf.pages)} pages")
                
            except Exception as e:
                print(f"  ⚠️ Processing failed for {filename}: {e}")
                print(f"  🔄 Falling back to standard processing...")
                # Fallback to standard processing
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                fallback_docs = loader.load()
                documents.extend(fallback_docs)
    
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Slightly larger chunks to preserve context
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        # Custom separators that respect content boundaries
        separators=[
            "\\n[/IMAGES]\\n",        # Don't split image sections
            "\\n[/TABLE]\\n",         # Don't split table sections  
            "\\n[/POTENTIAL_TABLE_SECTION]\\n",  # Don't split potential tables
            "\\n\\n",                # Paragraph breaks
            "\\n",                   # Line breaks
            ". ",                    # Sentence breaks
            " ",                     # Word breaks
        ]
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        
        # Show summary of enhanced content
        enhanced_chunks = [c for c in new_chunks if c.metadata.get('processing_type') == 'enhanced']
        if enhanced_chunks:
            total_images = sum(c.metadata.get('images_found', 0) for c in enhanced_chunks)
            total_extracted = sum(c.metadata.get('images_extracted', 0) for c in enhanced_chunks)
            total_tables = sum(c.metadata.get('tables_found', 0) for c in enhanced_chunks)
            print(f"📊 Content summary:")
            print(f"   🖼️ Total images detected: {total_images}")
            print(f"   💾 Total images extracted: {total_extracted}")
            print(f"   📋 Total tables detected: {total_tables}")
            if total_extracted > 0:
                print(f"   📁 Images saved to: {os.path.abspath(IMAGES_PATH)}")
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
