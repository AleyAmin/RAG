import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import shutil
from datetime import datetime
import json

# Import your existing modules
from rag_system import query_rag
from database import main as build_database, clear_database
from embedding_function import embedding_function
from langchain_community.vectorstores import Chroma

class RAGSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ezz Medical - RAG System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#1B7340',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        self.setup_styles()
        self.create_widgets()
        self.chat_history = []
        
    def setup_styles(self):
        """Configure custom styles for the application"""
        # Configure button styles
        self.style.configure('Primary.TButton', 
                           background=self.colors['primary'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Success.TButton',
                           background=self.colors['success'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Warning.TButton',
                           background=self.colors['warning'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Danger.TButton',
                           background=self.colors['danger'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        # Configure notebook style
        self.style.configure('TNotebook', background=self.colors['light'])
        self.style.configure('TNotebook.Tab', padding=[20, 10])
        
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header
        self.create_header(main_frame)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create tabs
        self.create_query_tab()
        self.create_database_tab()
        self.create_settings_tab()
        
    def create_header(self, parent):
        """Create the application header"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Logo/Title
        title_label = ttk.Label(header_frame, text="🏥 Ezz Medical RAG System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Status indicator
        self.status_frame = ttk.Frame(header_frame)
        self.status_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", 
                                     foreground=self.colors['success'])
        self.status_label.grid(row=0, column=0, padx=(0, 10))
        
        # Database status
        self.db_status_label = ttk.Label(self.status_frame, text="", 
                                        foreground=self.colors['primary'])
        self.db_status_label.grid(row=0, column=1)
        
        # Initialize database status after a short delay to ensure all widgets are created
        self.root.after(100, self.update_database_status)
        
    def create_query_tab(self):
        """Create the main query interface tab"""
        query_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(query_frame, text="💬 Ask Questions")
        
        # Configure grid weights
        query_frame.columnconfigure(0, weight=1)
        query_frame.rowconfigure(1, weight=1)
        
        # Query input section
        input_frame = ttk.LabelFrame(query_frame, text="Ask a Question", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Query entry
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(input_frame, textvariable=self.query_var, font=('Arial', 11))
        self.query_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.query_entry.bind('<Return>', lambda e: self.submit_query())
        
        # Submit button
        submit_btn = ttk.Button(input_frame, text="Ask", style='Primary.TButton',
                               command=self.submit_query)
        submit_btn.grid(row=0, column=1)
        
        # Sample questions section
        samples_frame = ttk.LabelFrame(input_frame, text="Sample Questions (Click to Use)", padding="5")
        samples_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        samples_frame.columnconfigure(0, weight=1)
        
        # Sample questions
        self.sample_questions = [
            "What is the accuracy of BERT embeddings for redundancy detection?",
            "How does the system detect redundancy in software requirements?",
            "What are the main evaluation metrics used in this study?",
            "Summarize the methodology used for requirement analysis",
            "What are the advantages of using BERT over traditional methods?",
            "How many documents were used in the evaluation dataset?",
            "What is the precision and recall of the proposed system?",
            "Compare the performance with other state-of-the-art methods"
        ]
        
        # Create buttons for sample questions (2 columns)
        samples_inner_frame = ttk.Frame(samples_frame)
        samples_inner_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        samples_inner_frame.columnconfigure(0, weight=1)
        samples_inner_frame.columnconfigure(1, weight=1)
        
        for i, question in enumerate(self.sample_questions):
            # Truncate long questions for button display
            display_text = question if len(question) <= 65 else question[:62] + "..."
            
            btn = ttk.Button(samples_inner_frame, text=display_text,
                           command=lambda q=question: self.use_sample_question(q))
            
            row = i // 2
            col = i % 2
            btn.grid(row=row, column=col, sticky=(tk.W, tk.E), padx=(0, 5) if col == 0 else (5, 0), pady=2)
        
        # Add helpful tip
        tip_label = ttk.Label(samples_frame, text="💡 Tip: Click any sample question to fill the input field, then press Ask",
                             font=('Arial', 8), foreground='gray')
        tip_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Advanced options
        advanced_frame = ttk.Frame(input_frame)
        advanced_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Number of results
        ttk.Label(advanced_frame, text="Results:").grid(row=0, column=0, sticky=tk.W)
        self.results_var = tk.IntVar(value=5)
        results_spin = ttk.Spinbox(advanced_frame, from_=1, to=10, width=5, 
                                  textvariable=self.results_var)
        results_spin.grid(row=0, column=1, padx=(5, 20), sticky=tk.W)
        
        # Clear chat button
        clear_btn = ttk.Button(advanced_frame, text="Clear Chat", 
                              command=self.clear_chat)
        clear_btn.grid(row=0, column=2, sticky=tk.E)
        
        # Chat history
        chat_frame = ttk.LabelFrame(query_frame, text="Chat History", padding="10")
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                                     font=('Arial', 10), state=tk.DISABLED)
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure chat tags for styling
        self.chat_display.tag_configure("user", foreground=self.colors['primary'], font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("assistant", foreground=self.colors['dark'])
        self.chat_display.tag_configure("sources", foreground=self.colors['secondary'], font=('Arial', 9, 'italic'))
        self.chat_display.tag_configure("timestamp", foreground='gray', font=('Arial', 8))
        
        # Export chat button
        export_frame = ttk.Frame(query_frame)
        export_frame.grid(row=2, column=0, sticky=tk.E, pady=(10, 0))
        
        export_btn = ttk.Button(export_frame, text="Export Chat", 
                               command=self.export_chat)
        export_btn.grid(row=0, column=0)
        
    def create_database_tab(self):
        """Create the database management tab"""
        db_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(db_frame, text="🗄️ Database")
        
        # Configure grid weights
        db_frame.columnconfigure(0, weight=1)
        db_frame.rowconfigure(2, weight=1)
        
        # Database info
        info_frame = ttk.LabelFrame(db_frame, text="Database Information", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        # Database stats
        self.db_info_text = tk.Text(info_frame, height=3, font=('Arial', 9), state=tk.DISABLED)
        self.db_info_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Refresh info button
        refresh_btn = ttk.Button(info_frame, text="Refresh Info", 
                                command=self.update_database_status)
        refresh_btn.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # Document management
        doc_frame = ttk.LabelFrame(db_frame, text="Document Management", padding="10")
        doc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons frame
        btn_frame = ttk.Frame(doc_frame)
        btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add documents button
        add_docs_btn = ttk.Button(btn_frame, text="📁 Add Documents", 
                                 style='Success.TButton',
                                 command=self.add_documents)
        add_docs_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Rebuild database button
        rebuild_btn = ttk.Button(btn_frame, text="🔄 Rebuild Database", 
                                style='Warning.TButton',
                                command=self.rebuild_database)
        rebuild_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Progress and logs
        log_frame = ttk.LabelFrame(db_frame, text="Process Logs", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Log display
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                    font=('Consolas', 9), state=tk.DISABLED)
        self.log_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(log_frame, variable=self.progress_var, 
                                           mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # API Configuration
        api_frame = ttk.LabelFrame(settings_frame, text="API Configuration", padding="10")
        api_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        api_frame.columnconfigure(1, weight=1)
        
        # API Key status
        ttk.Label(api_frame, text="Gemini API Key:").grid(row=0, column=0, sticky=tk.W)
        
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if api_key:
            key_status = f"Configured ({'*' * 4}{api_key[-4:]})"
            key_color = self.colors['success']
        else:
            key_status = "Not configured"
            key_color = self.colors['danger']
            
        api_status_label = ttk.Label(api_frame, text=key_status, foreground=key_color)
        api_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Paths configuration
        paths_frame = ttk.LabelFrame(settings_frame, text="Paths", padding="10")
        paths_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        paths_frame.columnconfigure(1, weight=1)
        
        ttk.Label(paths_frame, text="Documents folder:").grid(row=0, column=0, sticky=tk.W)
        content_path = os.path.abspath("content")
        ttk.Label(paths_frame, text=content_path).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(paths_frame, text="Database folder:").grid(row=1, column=0, sticky=tk.W)
        chroma_path = os.path.abspath("chroma")
        ttk.Label(paths_frame, text=chroma_path).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # About
        about_frame = ttk.LabelFrame(settings_frame, text="About", padding="10")
        about_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        about_text = """Ezz Medical RAG System v1.0
        
A Retrieval-Augmented Generation system for medical documents.
Uses Google's Gemini AI with Chroma vector database for intelligent
document search and question answering.

Features:
• Enhanced PDF processing with table and image detection
• Intelligent text chunking
• Vector similarity search
• Context-aware AI responses"""
        
        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT)
        about_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
    def submit_query(self):
        """Submit a query to the RAG system"""
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a question.")
            return
            
        # Check if database exists
        if not os.path.exists("chroma"):
            messagebox.showerror("Error", "Database not found. Please build the database first.")
            return
            
        # Clear query entry
        self.query_var.set("")
        
        # Add user message to chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_to_chat(f"[{timestamp}] You: {query}", "user")
        
        # Update status
        self.update_status("Processing query...", 'warning')
        
        # Start query in separate thread
        thread = threading.Thread(target=self._process_query, args=(query,))
        thread.daemon = True
        thread.start()
        
    def use_sample_question(self, question):
        """Use a sample question by setting it in the query field"""
        self.query_var.set(question)
        # Optionally auto-submit the query
        # self.submit_query()
        
    def _process_query(self, query):
        """Process query in background thread"""
        try:
            # Query the RAG system
            response = query_rag(query)
            
            # Update UI in main thread
            self.root.after(0, self._handle_query_response, response, query)
            
        except Exception as e:
            self.root.after(0, self._handle_query_error, str(e))
            
    def _handle_query_response(self, response, query):
        """Handle successful query response"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add response to chat
        self.add_to_chat(f"[{timestamp}] Assistant: {response.content}", "assistant")
        
        # Add to chat history
        self.chat_history.append({
            'timestamp': timestamp,
            'query': query,
            'response': response.content
        })
        
        self.update_status("Ready", 'success')
        
    def _handle_query_error(self, error):
        """Handle query error"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_to_chat(f"[{timestamp}] Error: {error}", "sources")
        self.update_status("Error occurred", 'danger')
        
    def add_to_chat(self, text, tag):
        """Add text to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text + "\n\n", tag)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def clear_chat(self):
        """Clear the chat history"""
        if messagebox.askyesno("Confirm", "Clear chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.chat_history.clear()
            
    def export_chat(self):
        """Export chat history to file"""
        if not self.chat_history:
            messagebox.showinfo("Info", "No chat history to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        for entry in self.chat_history:
                            f.write(f"[{entry['timestamp']}]\n")
                            f.write(f"Q: {entry['query']}\n")
                            f.write(f"A: {entry['response']}\n\n")
                            
                messagebox.showinfo("Success", f"Chat history exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
                
    def add_documents(self):
        """Add documents to the content folder"""
        files = filedialog.askopenfilenames(
            title="Select PDF documents",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if files:
            content_dir = "content"
            if not os.path.exists(content_dir):
                os.makedirs(content_dir)
                
            copied_files = []
            for file in files:
                try:
                    filename = os.path.basename(file)
                    dest_path = os.path.join(content_dir, filename)
                    shutil.copy2(file, dest_path)
                    copied_files.append(filename)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to copy {file}: {e}")
                    
            if copied_files:
                files_text = "\n".join(copied_files)
                messagebox.showinfo("Success", f"Added {len(copied_files)} file(s):\n{files_text}")
                self.log(f"Added {len(copied_files)} documents to content folder")
                
    def rebuild_database(self):
        """Rebuild the entire database"""
        if not messagebox.askyesno("Confirm", 
                                  "This will rebuild the entire database. Continue?"):
            return
            
        self.update_status("Rebuilding database...", 'warning')
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._rebuild_database)
        thread.daemon = True
        thread.start()
        
    def _rebuild_database(self):
        """Rebuild database in background thread"""
        try:
            self.root.after(0, self.log, "Starting database rebuild...")
                
            # Rebuild database
            build_database()
            
            self.root.after(0, self._rebuild_complete)
            
        except Exception as e:
            self.root.after(0, self._rebuild_error, str(e))
            
    def _rebuild_complete(self):
        """Handle successful database rebuild"""
        self.progress_bar.stop()
        self.update_status("Database rebuilt successfully", 'success')
        self.log("Database rebuild completed")
        self.update_database_status()
        
    def _rebuild_error(self, error):
        """Handle database rebuild error"""
        self.progress_bar.stop()
        self.update_status("Database rebuild failed", 'danger')
        self.log(f"Database rebuild failed: {error}")
        
    def clear_database_confirm(self):
        """Confirm and clear the database"""
        if messagebox.askyesno("Confirm", 
                              "This will permanently delete the database. Continue?"):
            try:
                clear_database()
                self.log("Database cleared")
                self.update_status("Database cleared", 'warning')
                self.update_database_status()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear database: {e}")
                
    def update_database_status(self):
        """Update database status information"""
        try:
            if os.path.exists("chroma"):
                # Try to get database info
                embeddings = embedding_function()
                db = Chroma(persist_directory="chroma", embedding_function=embeddings)
                
                # Get document count
                try:
                    existing_items = db.get(include=[])
                    doc_count = len(existing_items["ids"]) if existing_items["ids"] else 0
                    
                    self.db_status_label.config(text=f"📊 {doc_count} documents")
                    
                    # Update detailed info
                    content_files = []
                    if os.path.exists("content"):
                        content_files = [f for f in os.listdir("content") if f.endswith('.pdf')]
                    
                    info_text = f"""Database Status: ✅ Active
Documents in database: {doc_count}
PDF files in content folder: {len(content_files)}
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
                    
                    self.db_info_text.config(state=tk.NORMAL)
                    self.db_info_text.delete(1.0, tk.END)
                    self.db_info_text.insert(1.0, info_text)
                    self.db_info_text.config(state=tk.DISABLED)
                    
                except Exception:
                    self.db_status_label.config(text="❌ Database error")
                    self.db_info_text.config(state=tk.NORMAL)
                    self.db_info_text.delete(1.0, tk.END)
                    self.db_info_text.insert(1.0, "Database exists but may be corrupted")
                    self.db_info_text.config(state=tk.DISABLED)
            else:
                self.db_status_label.config(text="❌ No database")
                self.db_info_text.config(state=tk.NORMAL)
                self.db_info_text.delete(1.0, tk.END)
                self.db_info_text.insert(1.0, "Database not found. Please build the database first.")
                self.db_info_text.config(state=tk.DISABLED)
                
        except Exception as e:
            self.db_status_label.config(text="❌ Error")
            self.db_info_text.config(state=tk.NORMAL)
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(1.0, f"Error checking database: {e}")
            self.db_info_text.config(state=tk.DISABLED)
            
    def log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, log_message)
        self.log_display.config(state=tk.DISABLED)
        self.log_display.see(tk.END)
        
    def update_status(self, message, status_type='info'):
        """Update status label"""
        color = self.colors.get(status_type, self.colors['dark'])
        self.status_label.config(text=message, foreground=color)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = RAGSystemGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
