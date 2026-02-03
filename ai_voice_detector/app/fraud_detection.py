"""
Fraud Detection Module
Analyzes audio patterns, keywords, and behaviors to detect spam and fraud calls.
"""
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import librosa


@dataclass
class FraudAlert:
    """Represents a fraud detection alert"""
    is_fraud: bool
    confidence: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detected_patterns: List[str]
    fraud_indicators: Dict[str, float]
    recommended_action: str
    threat_type: Optional[str] = None


# Fraud keyword patterns (expandable)
FRAUD_KEYWORDS = {
    "financial_scam": [
        r'\b(bank account|credit card|debit card|cvv|pin code|pin|otp|one time password)\b',
        r'\b(transfer money|send money|payment|wire transfer|urgent payment)\b',
        r'\b(refund|tax refund|cashback|prize|lottery|won|winner)\b',
        r'\b(verify account|update account|suspended account|account blocked|bank blocked)\b',
        r'\b(kyc|know your customer|kyc update|kyc verification)\b',
    ],
    "impersonation": [
        r'\b(irs|internal revenue|tax department|social security)\b',
        r'\b(police|law enforcement|arrest warrant|legal action|court action)\b',
        r'\b(microsoft|apple|google|amazon|customer support)\b',
        r'\b(government official|federal agent|customs|immigration)\b',
    ],
    "urgency_tactics": [
        r'\b(urgent|urgently|immediately|right now|within \d+ (hours|minutes))\b',
        r'\b(last chance|final notice|act now|limited time|verify now)\b',
        r'\b(expire|expiring|expired|deadline)\b',
    ],
    "threat_coercion": [
        r'\b(arrest|sued|lawsuit|legal action|court|jail)\b',
        r'\b(penalty|fine|frozen account|suspended|suspend)\b',
        r'\b(serious consequences|take action against)\b',
    ],
    "information_request": [
        r'\b(social security number|ssn|date of birth|dob|password)\b',
        r'\b(credit card number|card details|banking information)\b',
        r'\b(confirm your|verify your|provide your|tell me your)\b',
        r'\b(aadhaar|aadhar|aadhaar number|pan|pan card|pan number)\b',
    ],
    "investment_scam": [
        r'\b(investment opportunity|guaranteed returns|double your money)\b',
        r'\b(crypto|bitcoin|trading|forex|stocks)\b',
        r'\b(risk free|no risk|100% profit|easy money)\b',
    ],
    "tech_support_scam": [
        r'\b(virus detected|computer infected|security alert|malware)\b',
        r'\b(remote access|teamviewer|anydesk|screen share)\b',
        r'\b(technician|tech support|technical team)\b',
    ],
}

# Suspicious phone number patterns
SPAM_NUMBER_PATTERNS = [
    r'^(800|888|877|866|855|844|833)',  # Toll-free numbers often used by telemarketers
    r'^\+1\s*(800|888|877|866|855|844|833)',
    r'^\d{10,}$',  # Very long numbers (international scams)
]


class FraudDetector:
    """Main fraud detection engine"""
    
    def __init__(self):
        self.fraud_keywords = FRAUD_KEYWORDS
        self.min_fraud_score = 0.3  # Threshold for fraud detection
        
    def analyze_audio(
        self,
        audio_path: str,
        transcription: Optional[str] = None,
        caller_number: Optional[str] = None,
        audio_features: Optional[np.ndarray] = None
    ) -> FraudAlert:
        """
        Comprehensive fraud analysis of audio call
        
        Args:
            audio_path: Path to audio file
            transcription: Transcribed text from audio
            caller_number: Phone number of caller
            audio_features: Pre-extracted audio features
        
        Returns:
            FraudAlert with detection results
        """
        fraud_scores = {}
        detected_patterns = []
        
        # 1. Analyze transcription for fraud keywords (PRIMARY DETECTION METHOD)
        if transcription:
            keyword_score, keyword_patterns = self._analyze_keywords(transcription)
            fraud_scores['keywords'] = keyword_score
            detected_patterns.extend(keyword_patterns)
        
        # Calculate overall fraud confidence
        # STRICT MODE: Keywords are the ONLY determining factor for fraud
        if fraud_scores.get('keywords', 0) > 0:
            # If keywords detected, it's fraud - use keyword score directly
            confidence = fraud_scores['keywords']
        else:
            # No fraud keywords = not fraud (0% confidence)
            confidence = 0.0
        
        # Determine risk level and recommended action
        is_fraud = confidence >= self.min_fraud_score
        risk_level = self._calculate_risk_level(confidence)
        recommended_action = self._get_recommended_action(risk_level, detected_patterns)
        threat_type = self._identify_threat_type(detected_patterns)
        
        return FraudAlert(
            is_fraud=is_fraud,
            confidence=confidence,
            risk_level=risk_level,
            detected_patterns=detected_patterns,
            fraud_indicators=fraud_scores,
            recommended_action=recommended_action,
            threat_type=threat_type
        )
    
    def _analyze_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Analyze text for fraud-related keywords - STRICT MODE: Only specific keywords trigger fraud detection"""
        if not text:
            return 0.0, []
            
        text_lower = text.lower()
        detected_patterns = []
        
        # STRICT FRAUD KEYWORDS - Only these keywords trigger fraud detection
        strict_fraud_keywords = [
            'otp',
            'aadhaar',
            'aadhar',  # Common misspelling
            'pan',
            'legal action',
            'suspend',
            'suspended',
            'urgent',
            'urgently',
            'police',
            'verify now',
            'payment',
            'lottery',
            'kyc',
            'bank blocked',
            'blocked',  # Also check "blocked" separately
            'account blocked',
        ]
        
        # Check for strict fraud keywords only
        keywords_found = []
        for keyword in strict_fraud_keywords:
            # Use word boundary for exact matches (case-insensitive)
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower, re.IGNORECASE):
                # Avoid duplicates (e.g., "blocked" and "account blocked")
                keyword_upper = keyword.upper()
                if keyword_upper not in keywords_found:
                    keywords_found.append(keyword_upper)
        
        # If NO keywords found, return 0 (not fraud)
        if not keywords_found:
            return 0.0, []
        
        # If ANY keyword found, mark as FRAUD with high confidence
        detected_patterns.append(f"🚨 FRAUD KEYWORDS DETECTED: {', '.join(keywords_found)}")
        
        # Calculate score based on number of keywords found
        # 1 keyword = 85%, 2+ keywords = 95-100%
        base_score = 0.85 + (min(len(keywords_found) - 1, 3) * 0.05)
        score = min(1.0, base_score)
        
        return score, detected_patterns
    
    def _analyze_audio_behavior(self, audio_path: str, features: Optional[np.ndarray]) -> Tuple[float, List[str]]:
        """Analyze audio behavioral patterns"""
        patterns = []
        score = 0.0
        
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30)
            
            # 1. Background noise analysis (robocalls often have consistent background)
            noise_level = np.std(y[:sr])  # First second
            if noise_level < 0.01:
                patterns.append("Suspiciously low background noise (possible robocall)")
                score += 0.2
            
            # 2. Speech rate analysis (scammers often speak rapidly)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if tempo > 140:  # Fast tempo
                patterns.append("Rapid speech pattern detected")
                score += 0.15
            
            # 3. Voice modulation (detect unnatural voice patterns)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_std = np.std(spectral_centroid)
            if spectral_std < 200:  # Low variation suggests synthetic voice
                patterns.append("Unnatural voice modulation")
                score += 0.2
            
            # 4. Silence detection (long pauses can indicate scripted calls)
            rms = librosa.feature.rms(y=y)[0]
            silence_threshold = 0.02
            silence_ratio = np.sum(rms < silence_threshold) / len(rms)
            if silence_ratio > 0.3:
                patterns.append("Unusual silence patterns")
                score += 0.1
            
            # 5. Audio quality (scammers often use poor quality VOIP)
            zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zero_crossings)
            if avg_zcr > 0.15 or avg_zcr < 0.03:
                patterns.append("Poor audio quality (VOIP/spoofed call)")
                score += 0.15
                
        except Exception as e:
            # If analysis fails, return neutral score
            patterns.append(f"Audio analysis warning: {str(e)}")
            score = 0.0
        
        return min(score, 1.0), patterns
    
    def _analyze_caller_number(self, number: str) -> Tuple[float, Optional[str]]:
        """Analyze caller number for spam patterns"""
        for pattern in SPAM_NUMBER_PATTERNS:
            if re.match(pattern, number):
                return 0.3, "Suspicious caller number pattern"
        return 0.0, None
    
    def _analyze_speech_patterns(self, audio_path: str) -> Tuple[float, List[str]]:
        """Analyze speech patterns for stress, urgency, aggression"""
        patterns = []
        score = 0.0
        
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30)
            
            # 1. Energy/intensity analysis (aggressive/urgent speech has high energy)
            rms_energy = librosa.feature.rms(y=y)[0]
            energy_variance = np.var(rms_energy)
            mean_energy = np.mean(rms_energy)
            
            if mean_energy > 0.15:
                patterns.append("High vocal intensity (possible aggression/urgency)")
                score += 0.15
            
            # 2. Pitch variation analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                pitch_std = np.std(pitch_values)
                if pitch_std > 100:  # High variation suggests stress/urgency
                    patterns.append("Stressed or urgent vocal patterns")
                    score += 0.2
            
            # 3. Speaking rate consistency
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            if len(onset_env) > 0:
                onset_std = np.std(onset_env)
                if onset_std > 1.5:
                    patterns.append("Inconsistent speaking rhythm")
                    score += 0.1
                    
        except Exception:
            score = 0.0
        
        return min(score, 1.0), patterns
    
    def _calculate_risk_level(self, confidence: float) -> str:
        """Calculate risk level based on confidence score"""
        if confidence >= 0.70:  # Lowered threshold for CRITICAL (was 0.75)
            return "CRITICAL"
        elif confidence >= 0.50:  # Lowered threshold for HIGH (was 0.55)
            return "HIGH"
        elif confidence >= 0.30:  # Lowered threshold for MEDIUM (was 0.35)
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommended_action(self, risk_level: str, patterns: List[str]) -> str:
        """Get recommended action based on risk level"""
        actions = {
            "CRITICAL": "⛔ BLOCK IMMEDIATELY - Do not engage. Hang up and report to authorities.",
            "HIGH": "⚠️ HIGH RISK - Hang up immediately. Do not provide any information.",
            "MEDIUM": "⚡ CAUTION - Be extremely careful. Verify caller identity independently.",
            "LOW": "ℹ️ Monitor call. Stay vigilant and don't share sensitive information."
        }
        return actions.get(risk_level, actions["LOW"])
    
    def _identify_threat_type(self, patterns: List[str]) -> Optional[str]:
        """Identify the type of fraud/threat"""
        pattern_text = " ".join(patterns).lower()
        
        if "financial" in pattern_text or "bank" in pattern_text:
            return "Financial Scam"
        elif "impersonation" in pattern_text or "government" in pattern_text:
            return "Government Impersonation"
        elif "tech support" in pattern_text or "virus" in pattern_text:
            return "Tech Support Scam"
        elif "investment" in pattern_text:
            return "Investment Fraud"
        elif "threat" in pattern_text or "coercion" in pattern_text:
            return "Threatening/Coercion"
        elif "robocall" in pattern_text:
            return "Automated Robocall"
        elif "urgency" in pattern_text:
            return "Urgency-Based Scam"
        
        return "Unknown Threat"


class RealTimeAlertSystem:
    """Real-time alert generation and notification system"""
    
    def __init__(self):
        self.alert_history = []
        self.blocked_numbers = set()
    
    def generate_alert(self, fraud_alert: FraudAlert, caller_info: Dict = None) -> Dict:
        """Generate a structured alert for the user"""
        alert = {
            "timestamp": self._get_timestamp(),
            "alert_type": "FRAUD_DETECTED" if fraud_alert.is_fraud else "SUSPICIOUS_ACTIVITY",
            "risk_level": fraud_alert.risk_level,
            "confidence": round(fraud_alert.confidence * 100, 1),
            "threat_type": fraud_alert.threat_type,
            "caller_info": caller_info or {},
            "detected_patterns": fraud_alert.detected_patterns,
            "fraud_indicators": {
                k: round(v * 100, 1) for k, v in fraud_alert.fraud_indicators.items()
            },
            "recommended_action": fraud_alert.recommended_action,
            "should_block": fraud_alert.confidence >= 0.75,
        }
        
        self.alert_history.append(alert)
        
        # Auto-block high-risk numbers
        if alert["should_block"] and caller_info and "number" in caller_info:
            self.blocked_numbers.add(caller_info["number"])
        
        return alert
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def is_blocked(self, number: str) -> bool:
        """Check if a number is blocked"""
        return number in self.blocked_numbers
    
    def block_number(self, number: str):
        """Manually block a number"""
        self.blocked_numbers.add(number)
    
    def unblock_number(self, number: str):
        """Unblock a number"""
        self.blocked_numbers.discard(number)
