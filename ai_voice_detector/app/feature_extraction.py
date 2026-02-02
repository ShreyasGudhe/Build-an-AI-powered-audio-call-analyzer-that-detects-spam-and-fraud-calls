import librosa
import numpy as np

def extract_features(audio_path, for_language=False):
    """Extract audio features.
    
    Args:
        audio_path: Path to audio file or file-like object
        for_language: If True, extract language-specific features
    """
    y, sr = librosa.load(audio_path, sr=None, duration=30 if for_language else None)

    # Basic features - ensure all are 1D arrays
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral = np.atleast_1d(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)).flatten()
    zcr = np.atleast_1d(np.mean(librosa.feature.zero_crossing_rate(y))).flatten()
    
    features = [mfcc, chroma, spectral, zcr]
    
    if for_language:
        # Add language-discriminative features
        # Mel-frequency cepstral coefficients statistics (better for language ID)
        mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc_full)
        mfcc_delta2 = librosa.feature.delta(mfcc_full, order=2)
        
        # Statistics of MFCCs and deltas - ensure all are 1D arrays
        features.append(np.mean(mfcc_full.T, axis=0))
        features.append(np.std(mfcc_full.T, axis=0))
        features.append(np.mean(mfcc_delta.T, axis=0))
        features.append(np.mean(mfcc_delta2.T, axis=0))
        
        # Spectral features (languages have different spectral characteristics)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        spectral_rolloff = np.atleast_1d(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)).flatten()
        features.append(spectral_contrast)
        features.append(spectral_rolloff)
        
        # Rhythm and prosody features (important for language identification)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(np.array([float(tempo)]))

    return np.hstack(features)
