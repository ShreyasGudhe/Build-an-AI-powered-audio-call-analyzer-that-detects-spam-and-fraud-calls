"""
Speech-to-Text Transcription Module
Converts audio to text for keyword and content analysis.
"""
import os
import io
from pathlib import Path
from typing import Optional, Dict
import numpy as np

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class SpeechTranscriber:
    """Transcribes audio to text using Whisper model"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       - tiny: Fastest, least accurate
                       - base: Good balance (recommended)
                       - small: Better accuracy
                       - medium/large: Best accuracy, slower
        """
        self.model = None
        self.model_size = model_size
        
        if WHISPER_AVAILABLE:
            try:
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            except Exception as e:
                print(f"Warning: Could not load Whisper model: {e}")
                self.model = None
        else:
            print("Warning: faster-whisper not available. Install it for transcription support.")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        max_duration: float = 30.0
    ) -> Dict[str, any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'hi', 'ta') or None for auto-detect
            max_duration: Maximum audio duration to transcribe (seconds)
        
        Returns:
            Dictionary with transcription results:
            - text: Full transcribed text
            - segments: List of transcribed segments with timestamps
            - language: Detected or specified language
            - confidence: Average confidence score
        """
        if not self.model:
            return {
                "text": "",
                "segments": [],
                "language": "unknown",
                "confidence": 0.0,
                "error": "Transcription model not available"
            }
        
        try:
            # Transcribe with Whisper
            segments_list = []
            full_text = []
            confidence_scores = []
            
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                )
            )
            
            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": segment.avg_logprob
                })
                full_text.append(segment.text.strip())
                confidence_scores.append(segment.avg_logprob)
                
                # Stop if we exceed max duration
                if segment.end > max_duration:
                    break
            
            # Calculate average confidence
            avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
            # Convert log probability to approximate confidence (0-1 scale)
            # Whisper returns log probabilities (negative values)
            avg_confidence = max(0.0, min(1.0, 1.0 + (avg_confidence / 2.0)))
            
            return {
                "text": " ".join(full_text),
                "segments": segments_list,
                "language": info.language if hasattr(info, 'language') else "unknown",
                "confidence": avg_confidence
            }
            
        except Exception as e:
            return {
                "text": "",
                "segments": [],
                "language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def transcribe_realtime(
        self,
        audio_chunk: bytes,
        sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Transcribe audio chunk in real-time
        
        Args:
            audio_chunk: Audio data as bytes
            sample_rate: Audio sample rate
        
        Returns:
            Transcribed text or None if failed
        """
        if not self.model:
            return None
        
        try:
            # Create temporary file-like object
            audio_io = io.BytesIO(audio_chunk)
            
            segments, _ = self.model.transcribe(audio_io, beam_size=1)  # Faster for real-time
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            return " ".join(text_parts)
            
        except Exception:
            return None


class KeywordDetector:
    """Fast keyword detection for real-time analysis"""
    
    def __init__(self, keywords: Dict[str, list]):
        """
        Initialize keyword detector
        
        Args:
            keywords: Dictionary mapping category names to lists of keywords/patterns
        """
        self.keywords = keywords
    
    def detect_in_text(self, text: str) -> Dict[str, list]:
        """
        Detect keywords in text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary mapping categories to detected keywords
        """
        import re
        
        text_lower = text.lower()
        detected = {}
        
        for category, patterns in self.keywords.items():
            matches = []
            for pattern in patterns:
                if isinstance(pattern, str) and pattern.startswith('\\b'):
                    # Regex pattern
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        matches.append(pattern)
                else:
                    # Simple keyword
                    if pattern.lower() in text_lower:
                        matches.append(pattern)
            
            if matches:
                detected[category] = matches
        
        return detected
    
    def detect_in_segments(self, segments: list) -> list:
        """
        Detect keywords in timestamped segments
        
        Args:
            segments: List of segment dictionaries with 'text', 'start', 'end'
        
        Returns:
            List of detected keywords with timestamps
        """
        detections = []
        
        for segment in segments:
            detected = self.detect_in_text(segment['text'])
            if detected:
                detections.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'],
                    "detected_categories": list(detected.keys()),
                    "matches": detected
                })
        
        return detections


# Convenience function for quick transcription
def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
    """
    Quick transcription helper
    
    Args:
        audio_path: Path to audio file
        language: Optional language code
    
    Returns:
        Transcribed text
    """
    transcriber = SpeechTranscriber()
    result = transcriber.transcribe(audio_path, language=language)
    return result.get("text", "")
