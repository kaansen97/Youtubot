# src/services/rag_service.py 

from typing import Optional, List, Dict, Any, Tuple
import re
from pathlib import Path
import sys
import os
import json
import shutil
import whisper

# --- Add src to path for imports ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# --- Imports ---
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langdetect import detect, LangDetectException
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import yt_dlp

from core.models import AppConfig, SearchResult, RAGResponse
from core.config import get_config, get_prompts
from services.web_search_service import WebSearchService

# --- Constants ---
LANGUAGE_NAME_MAP = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "tr": "Turkish"
}

class RAGService:
    def __init__(self):
        """Initialize the RAG service with components and configuration."""
        self.config: AppConfig = get_config()
        self.prompts: Dict[str, Any] = get_prompts()
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = ChatOllama(model=self.config.llm_model_name, base_url=ollama_host)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            show_progress=False
        )
        
        # --- Whisper Model Initialization ---
        try:
            # 'base' is multilingual, 'base.en' is English-only and faster.
            self.whisper_model = whisper.load_model("base") 
            print("Whisper model 'base' loaded successfully.")
        except Exception as e:
            self.whisper_model = None
            print(f"Warning: Could not load Whisper model: {e}. Local transcription will be unavailable.")
        
        self.vector_store: Optional[FAISS] = None
        self.processed_videos_metadata: List[Dict[str, Any]] = []
        self.web_search_service = WebSearchService()
        self.confidence_threshold = 0.5
        
        self.db_base_path = Path(self.config.vector_db_path)
        self.db_base_path.mkdir(parents=True, exist_ok=True)
        # Also ensure a data path for temporary audio files
        (self.db_base_path.parent / "temp").mkdir(exist_ok=True)

    def _transcribe_with_whisper(self, url: str) -> Optional[str]:
        """Transcribes audio from a YouTube URL using Whisper. SLOW."""
        if not self.whisper_model:
            print("Whisper model not available.")
            return None
            
        try:
            temp_audio_path = self.db_base_path.parent / "temp/temp_audio.mp3"
            
            # yt-dlp options to download audio only
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(temp_audio_path).replace('.mp3', ''), # yt-dlp adds extension
                'nocheckcertificate': True,
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            result = self.whisper_model.transcribe(str(temp_audio_path), fp16=False)
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return result['text']
            
        except Exception as e:
            print(f"Whisper transcription failed for {url}: {e}")
            if os.path.exists(temp_audio_path):
                 os.remove(temp_audio_path) 
            return None

    def _get_video_docs_and_meta(self, url: str, lang_code: str, use_whisper: bool = False) -> Optional[Tuple[List[Document], Dict]]:
        """Helper to get docs. Tries API first, then falls back to Whisper if requested."""
        try:
            ydl_opts = {'quiet': True, 'skip_download': True, 'nocheckcertificate': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            video_id = info.get("id")
            metadata = {
                'title': info.get('title', 'Unknown Title'),
                'source': f"https://www.youtube.com/watch?v={video_id}",
                'author': info.get('uploader', 'Unknown Author'),
                'language': lang_code
            }
            
            transcript_text = None
            if not use_whisper:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_transcript([lang_code, 'en'])
                    transcript_text = " ".join([chunk.text for chunk in transcript.fetch()])
                    print(f"Successfully fetched transcript for '{metadata['title']}' via API.")
                except (TranscriptsDisabled, NoTranscriptFound):
                    print(f"API transcript not found for '{metadata['title']}'. Whisper fallback is available if selected.")

            if transcript_text is None and use_whisper:
                print(f"Using Whisper to transcribe '{metadata['title']}'. This may take a while...")
                transcript_text = self._transcribe_with_whisper(url)
            
            if not transcript_text:
                print(f"Warning: Skipping video {url} - No transcript could be obtained.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            doc = Document(page_content=transcript_text, metadata=metadata)
            docs = text_splitter.split_documents([doc])
            
            return docs, metadata
        except Exception as e:
            print(f"Warning: A critical error occurred while processing {url}: {e}")
            return None

    def process_content(self, content_url: str, lang_code: str, content_type: str = "video", use_whisper: bool = False):
        """Processes a single video or a whole playlist and creates a vector store in memory."""
        all_docs = []
        self.processed_videos_metadata = []

        video_urls = []
        if content_type == 'playlist':
            print(f"Processing playlist: {content_url}")
            ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(content_url, download=False)
                video_urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in playlist_info.get('entries', [])]
        else:
            video_urls.append(content_url)

        for i, url in enumerate(video_urls):
            if content_type == 'playlist':
                print(f"Processing video {i+1}/{len(video_urls)}: {url}")
            result = self._get_video_docs_and_meta(url, lang_code, use_whisper)
            if result:
                docs, meta = result
                all_docs.extend(docs)
                self.processed_videos_metadata.append(meta)

        if not all_docs:
            print("Error: No documents were processed.")
            return False
            
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        print("In-memory vector store created successfully.")
        return True

    def save_index_to_disk(self, session_name: str):
        """Saves the current in-memory vector store and metadata to disk."""
        if not self.vector_store or not session_name:
            return
        session_path = self.db_base_path / session_name
        session_path.mkdir(exist_ok=True)
        self.vector_store.save_local(str(session_path))
        with open(session_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.processed_videos_metadata, f, ensure_ascii=False, indent=4)
        print(f"Session '{session_name}' saved to {session_path}")

    def load_index_from_disk(self, session_name: str) -> bool:
        """Loads a vector store and its metadata from disk into memory."""
        session_path = self.db_base_path / session_name
        if not session_path.exists():
            return False
        try:
            self.vector_store = FAISS.load_local(str(session_path), self.embeddings, allow_dangerous_deserialization=True)
            with open(session_path / "metadata.json", "r", encoding="utf-8") as f:
                self.processed_videos_metadata = json.load(f)
            print(f"Session '{session_name}' loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading session '{session_name}': {e}")
            return False

    def list_saved_sessions(self) -> List[str]:
        """Returns a list of all saved session names."""
        if not self.db_base_path.exists():
            return []
        return [d.name for d in self.db_base_path.iterdir() if d.is_dir()]

    def delete_session(self, session_name: str) -> bool:
        """Deletes a saved session from the disk."""
        session_path = self.db_base_path / session_name
        if not session_path.exists() or not session_path.is_dir():
            return False
        try:
            shutil.rmtree(session_path)
            print(f"Session '{session_name}' deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting session '{session_name}': {e}")
            return False

    def generate_response(self, query: str, override_language: Optional[str] = None) -> RAGResponse:
        if not self.vector_store:
            return RAGResponse(query=query, answer="Please process a video/playlist or load a session first.", sources=[], language="en")
        
        base_language = self.processed_videos_metadata[0].get('language', 'en') if self.processed_videos_metadata else 'en'
        final_language = override_language if override_language else base_language
        
        relevant_docs = self.vector_store.similarity_search_with_score(query, k=self.config.retrieval_k)
        
        if not relevant_docs:
            return self._web_search_fallback(query, base_language, override_language)
        
        context = "\n---\n".join([doc.page_content for doc, score in relevant_docs])
        rag_answer = self._generate_answer(query, context, base_language, 'rag_prompt', override_language)
        
        confidence = relevant_docs[0][1] if relevant_docs else 0.0
        
        if confidence >= self.confidence_threshold:
            search_results = [
                SearchResult(
                    video_title=doc.metadata.get("title", ""),
                    video_url=doc.metadata.get("source", ""),
                    text_content=doc.page_content,
                    similarity_score=score
                ) for doc, score in relevant_docs
            ]
            return RAGResponse(query=query, answer=rag_answer, sources=search_results, confidence_score=confidence, language=final_language)
        else:
            return self._web_search_fallback(query, base_language, override_language)

    def _generate_answer(self, question: str, context: str, base_language: str, prompt_key: str, override_language: Optional[str] = None) -> str:
        prompt_templates = self.prompts.get(prompt_key)
        template_string = prompt_templates.get(base_language, prompt_templates.get("en"))
        formatted_prompt = template_string.format(context=context, question=question)

        if override_language:
            lang_name = LANGUAGE_NAME_MAP.get(override_language, override_language)
            override_instruction = f"\n\nIMPORTANT: You must provide the final answer in the following language: {lang_name}."
            formatted_prompt += override_instruction
        
        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()

    def _web_search_fallback(self, query: str, base_language: str, override_language: Optional[str] = None) -> RAGResponse:
        final_language = override_language if override_language else base_language
        web_result = self.web_search_service.search(query)
        
        if web_result and web_result.snippet:
            web_answer = self._generate_answer(query, web_result.snippet, base_language, 'web_qa_prompt', override_language)
            confidence = self._evaluate_response_quality(query, web_answer)
            source = SearchResult(video_title=f"Web Search: {web_result.title}", video_url=web_result.url, text_content=web_result.snippet, similarity_score=0.0)
            return RAGResponse(query=query, answer=web_answer, sources=[source], confidence_score=confidence, language=final_language)
        else:
            no_content_message = self.prompts.get('no_context_prompt', {}).get(final_language, "Content not found.")
            return RAGResponse(query=query, answer=no_content_message, sources=[], confidence_score=0.0, language=final_language)

    def _evaluate_response_quality(self, query: str, response: str) -> float:
        try:
            eval_prompt_template = self.prompts.get('evaluation_prompt')
            if not eval_prompt_template: return 0.5
            formatted_prompt = eval_prompt_template.format(query=query, response=response)
            eval_response = self.llm.invoke(formatted_prompt)
            match = re.search(r"(\d\.\d+)", eval_response.content)
            return float(match.group(1)) if match else 0.5
        except Exception as e:
            print(f"Could not evaluate response quality: {e}")
            return 0.5