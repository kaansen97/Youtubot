# src/core/models.py (Upgraded for Multilingual & Open-Source LLM)

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# --- Data Models (These are well-structured and remain the same) ---

@dataclass
class SearchResult:
    """Search result from vector database."""
    video_title: str
    video_url: str
    text_content: str
    similarity_score: float

@dataclass
class RAGResponse:
    """RAG system response, now with language tracking."""
    query: str
    answer: str
    sources: List[SearchResult]
    language: str  # To track language for TTS service.
    confidence_score: Optional[float] = None


@dataclass
class AppConfig:
    """
    Application configuration for a dynamic, multilingual RAG system.
    Values are loaded from settings.yaml and environment variables.
    """
    # --- API Settings (from environment variables) ---
    # We still use OpenAI for its top-tier, low-cost embedding models.
    # The LLM itself will be local (Ollama) and won't require a key.
    openai_api_key: str = ""
    elevenlabs_api_key: Optional[str] = None

    # --- Model Configuration (from settings.yaml) ---
    # This will now be the name of the local Ollama model (e.g., "llama3")
    llm_model_name: str = "llama3"
    # OpenAI's embedding models are highly performant and multilingual.
    embedding_model: str = "text-embedding-3-small"
    
    # --- RAG Settings (from settings.yaml) ---
    retrieval_k: int = 4
    
    # --- ✨ NEW: TTS Service Settings (from settings.yaml) ---
    language_voice_map: Dict[str, str] = field(default_factory=dict)
    
    # --- File Paths (from settings.yaml) ---
    # We simplify this; we only need a place for the temporary vector store.
    data_dir: str = "data"
    vector_db_path: str = "data/vector_db_cache"

    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate that essential API keys are present."""
        # The primary key needed is for OpenAI Embeddings.
        if not self.openai_api_key:
            print("❌ Validation Error: OPENAI_API_KEY environment variable is required for embeddings!")
            return False
        
        # ElevenLabs key is optional; TTS will be disabled if it's missing.
        if not self.elevenlabs_api_key:
            print("⚠️ Validation Warning: ELEVENLABS_API_KEY is not set. TTS will be disabled.")
            
        print("✅ Configuration validated successfully.")
        return True