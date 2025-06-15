# services/tts.py (gTTS Online-Only Edition)

import io
from gtts import gTTS
from typing import Optional

class TTSService:
    """
    An online-only TTS service using Google's gTTS library.
    It's lightweight, reliable, and doesn't require local models or dependencies.
    The only requirement is an active internet connection.
    """
    
    def __init__(self):
        self.is_available_flag = True
        print("TTSService initialized using gTTS (online).")

    def is_available(self) -> bool:
        """
        For gTTS, availability simply means having an internet connection.
        This is checked implicitly when we try to generate speech.
        """
        return self.is_available_flag

    def generate_speech(self, text: str, language_code: str = "en") -> Optional[bytes]:
        """
        Generate speech from text using gTTS and return the audio data as bytes.
        This method requires an internet connection to function.
        """
        if not text:
            return None

        try:
            # 1. Create a gTTS object with the text and language
            tts_object = gTTS(text=text, lang=language_code, slow=False)
            
            audio_fp = io.BytesIO()
            
            tts_object.write_to_fp(audio_fp)
            
            audio_fp.seek(0)
            
            audio_data = audio_fp.read()
            return audio_data
            
        except Exception as e:

            print(f"Error during gTTS generation: {e}")
            print("This may be due to a lack of internet connection or an invalid language code.")

            self.is_available_flag = False 
            return None