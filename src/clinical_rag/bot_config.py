import sqlite3
import logging
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            current_lang TEXT DEFAULT 'en',
            custom_prompt TEXT
        )
    ''')
    
    conn.commit()
    return conn

# Language settings
LANGUAGES = {
    'en': {
        'welcome': """<b>- Welcome to Article Assistant!</b>

This bot uses RAG (Retrieval-Augmented Generation) to answer questions based on your documents.

<b>- How it works:</b>
1. Upload a file (PDF/TXT)
2. Ask questions about the content
3. Get AI-powered answers

<b>- Take part in our project:</b>
- <a href="https://github.com/Konstantin-vanov-hub/Article-Assistant--RAG-Telegram-Bot">GitHub Repository</a>
- Developer: <a href="https://t.me/Konstantin_vanov">@Konstantin_vanov</a>

<b>Please upload your first document to begin!</b>
""",
        'ask_btn': "Ask question",
        'article_btn': "Enter article",
        'lang_btn': "Change language",
        'prompt_btn': "Prompt settings",
        'summarize_btn': "Summary",
        'lang_changed': "Language changed to English",
        'ask_prompt': "📝 Please enter your question about the article:",
        'processing': "🔍 Searching for answer in the article...",
        'indexing': "📚 Indexing article content...",
        'summarizing': "📝 Generating summary...",
        'summary_title': "📌 Main Points:",
        'no_content': "No content to summarize",
        'after_answer': "💡 You can ask another question or choose another option below",
        'cancel': "Cancel",
        'error': "❌ Error occurred",
        'enter_url': "🌐 Please enter the article URL:",
        'no_article_error': "⚠️ Please add an article first using the 'Enter article' button",
        'invalid_input': "⚠️ Please use the buttons below to interact with me",
        'index_success': "✅ Article indexed successfully!",
        'chunks_info': "Processed {} content chunks.\n\nYou can now ask questions about this article.",
        'prompt_menu': "✏️ Choose prompt option:",
        'default_prompt': "Use default prompt",
        'custom_prompt': "Write new prompt",
        'enter_custom_prompt': "📝 Enter your custom prompt (e.g., 'Answer in technical style'):",
        'prompt_saved': "✅ Custom prompt saved! Now ask your question.",
        'current_prompt': "Current prompt: {}",
        'setup_guide': "🔧 Setup guide: https://github.com/Konstantin-vanov-hub/Article-Assistant--RAG-Telegram-Bot#setup",
        'invalid_url': "⚠️ Please enter a valid URL starting with http:// or https://",
        'file_too_large': "⚠️ File is too large (max 10MB). Please upload a smaller file.",
        'file_empty': "⚠️ The file is empty. Please upload a valid file with content.",
        'unsupported_format': "⚠️ Unsupported file format. Please upload PDF or TXT files only.",
        'file_uploaded': "📥 File uploaded successfully! Processing...",
        'file_processed': "File processed",
        'url_processed': "URL processed successfully",
        'file_not_found': "❌ File not found. Please try uploading again.",
        'no_content_found': "❌ No readable content found in the document. Please try another file.",
        'connection_error': "❌ Connection error. Please check your internet connection and try again.",
        'api_key_error': "❌ System error: API key configuration issue. Please contact administrator.",
        'processing_file': "🔍 Processing your file...",
        'document_too_large': "⚠️ Document is too large. Processing in parts...",
        'max_size_exceeded': "❌ Document exceeds maximum processing size. Please use a smaller document.",
        'youtube_btn': "📹 Process YouTube Video",
        'enter_youtube_url': "🎥 Please enter a YouTube video URL:",
        'downloading_video': "⬇️ Downloading video...",
        'transcribing_video': "🎤 Transcribing video content...",
        'video_processed': "✅ Video processed successfully!",
        'video_title': "📺 Video: {}",
        'video_duration': "⏱️ Duration: {}",
        'transcript_length': "📝 Transcript length: {} characters",
        'invalid_youtube_url': "⚠️ Please enter a valid YouTube URL",
        'video_too_long': "⚠️ Video is too long (max 2 hours). Please choose a shorter video.",
        'transcription_failed': "❌ Failed to transcribe video. Please try another video.",
        'download_failed': "❌ Failed to download video. Please check the URL and try again.",
    },
   
    }
}

# Conversation states
(MAIN_MENU, ENTER_LINK, CHANGE_LANG, ASK_QUESTION, PROMPT_MENU, ENTER_CUSTOM_PROMPT, SUMMARIZE_DOC, ENTER_YOUTUBE_URL) = range(8)

DEFAULT_PROMPT = {
    'en': """Expert Research Assistant Guidelines:

1. Source Accuracy:
   - Strictly use ONLY the provided context
   - For missing info: "The article doesn't specify"
   - Never hallucinate facts

2. Response Structure:
   - Core Answer (1 bolded sentence)
   - Key Evidence (3-5 bullet points max)
   - Practical Implications (when relevant)
   - Limitations (if data is incomplete)

3. Technical Content:
   - Code: ```python\n...\n``` 
   - Formulas: $E=mc^2$ format
   - Terms: "API (Application Programming Interface)"

4. Language Rules:
   - Match question's language
   - Auto-correct grammar subtly
   - Use ISO standards for dates/units

Context:
{context}

Question: {question}""",

    'summary_prompt': {
        'en': """Generate a concise 3-5 bullet point summary in English focusing on:
        - Key arguments
        - Unique findings
        - Practical applications
        
        Text: {text}""",
        
    }
}