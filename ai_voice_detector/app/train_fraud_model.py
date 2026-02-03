"""
Fraud Detection Model Trainer
Trains machine learning models to detect fraud patterns in audio calls.
"""
import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from .feature_extraction import extract_features


class FraudModelTrainer:
    """Train models to detect fraudulent calls"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model to train
                       - "random_forest": Random Forest (recommended)
                       - "gradient_boost": Gradient Boosting (slower, potentially more accurate)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_dataset(
        self,
        fraud_audio_dir: str,
        legitimate_audio_dir: str,
        max_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training dataset from audio files
        
        Args:
            fraud_audio_dir: Directory containing fraud call recordings
            legitimate_audio_dir: Directory containing legitimate call recordings
            max_samples: Maximum samples per class (None = all)
        
        Returns:
            Tuple of (features, labels)
        """
        print("Extracting features from fraud calls...")
        fraud_features, fraud_labels = self._extract_features_from_dir(
            fraud_audio_dir, label=1, max_samples=max_samples
        )
        
        print("Extracting features from legitimate calls...")
        legit_features, legit_labels = self._extract_features_from_dir(
            legitimate_audio_dir, label=0, max_samples=max_samples
        )
        
        # Combine datasets
        X = np.vstack([fraud_features, legit_features])
        y = np.concatenate([fraud_labels, legit_labels])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"Dataset prepared: {len(X)} samples ({np.sum(y)} fraud, {len(y) - np.sum(y)} legitimate)")
        
        return X, y
    
    def _extract_features_from_dir(
        self,
        directory: str,
        label: int,
        max_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all audio files in a directory"""
        audio_dir = Path(directory)
        
        if not audio_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
            audio_files.extend(audio_dir.glob(f"**/*{ext}"))
        
        if max_samples:
            audio_files = audio_files[:max_samples]
        
        features_list = []
        labels_list = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                print(f"  Processing {i+1}/{len(audio_files)}: {audio_file.name}")
                features = extract_features(str(audio_file), for_language=False)
                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                print(f"  Error processing {audio_file.name}: {e}")
                continue
        
        if not features_list:
            raise ValueError(f"No valid audio files found in {directory}")
        
        return np.array(features_list), np.array(labels_list)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cross_validate: bool = True
    ) -> dict:
        """
        Train the fraud detection model
        
        Args:
            X: Feature matrix
            y: Labels (0=legitimate, 1=fraud)
            test_size: Proportion of data for testing
            cross_validate: Whether to perform cross-validation
        
        Returns:
            Dictionary with training results
        """
        print("\nTraining fraud detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        results = {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True
            ),
            "model_type": self.model_type,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test)
        }
        
        # Cross-validation
        if cross_validate:
            print("\nPerforming 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            results["cv_scores"] = cv_scores.tolist()
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        
        return results
    
    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data.get("model_type", "unknown")
        
        print(f"Model loaded from: {model_path}")
    
    def predict(self, audio_path: str) -> Tuple[int, float]:
        """
        Predict if an audio file contains fraud
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (prediction, confidence)
            - prediction: 0=legitimate, 1=fraud
            - confidence: Probability of fraud (0-1)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Extract features
        features = extract_features(audio_path, for_language=False)
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[1]  # Probability of fraud
        
        return int(prediction), float(confidence)


def train_fraud_model(
    fraud_dir: str,
    legitimate_dir: str,
    output_path: str = None,
    model_type: str = "random_forest"
):
    """
    Convenience function to train a fraud detection model
    
    Args:
        fraud_dir: Directory with fraud call recordings
        legitimate_dir: Directory with legitimate call recordings
        output_path: Where to save the model (default: models/fraud_model.pkl)
        model_type: Model type to train
    """
    # Default output path
    if output_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        output_path = base_dir / "models" / "fraud_model.pkl"
    
    # Initialize trainer
    trainer = FraudModelTrainer(model_type=model_type)
    
    # Prepare dataset
    X, y = trainer.prepare_dataset(fraud_dir, legitimate_dir)
    
    # Train model
    results = trainer.train(X, y, cross_validate=True)
    
    # Save model
    trainer.save_model(str(output_path))
    
    return trainer, results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python train_fraud_model.py <fraud_dir> <legitimate_dir> [output_path]")
        sys.exit(1)
    
    fraud_dir = sys.argv[1]
    legitimate_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    trainer, results = train_fraud_model(fraud_dir, legitimate_dir, output_path)
    
    print("\n=== Training Complete ===")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
