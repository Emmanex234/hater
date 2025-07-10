import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import pickle
import logging
from threading import Thread
import time
import unicodedata
from datetime import datetime
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Simple HTML template since we don't have a templates folder
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Hate Speech Detection API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { background: #007bff; color: white; padding: 3px 8px; border-radius: 3px; }
        code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced Hate Speech Detection API</h1>
        <p>This API uses machine learning to detect hate speech in text with high accuracy.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Predict if a text contains hate speech</p>
            <p><strong>Body:</strong> <code>{"text": "your text here"}</code></p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict/batch</h3>
            <p>Predict multiple texts at once</p>
            <p><strong>Body:</strong> <code>{"texts": ["text1", "text2"]}</code></p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /train</h3>
            <p>Train the model with new data</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /status</h3>
            <p>Check model training status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health</p>
        </div>
        
        <h2>Example Usage:</h2>
        <pre>
curl -X POST {{ base_url }}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "I hate this weather"}'
        </pre>
    </div>
</body>
</html>
"""

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Error in {func.__name__}: {error}")
            
        end_time = time.time()
        
        # Log metrics
        logger.info(f"Function: {func.__name__}")
        logger.info(f"Duration: {end_time - start_time:.4f}s")
        logger.info(f"Success: {success}")
        
        if success:
            return result
        else:
            raise Exception(error)
    return wrapper

class OptimizedHateSpeechDetector:
    def __init__(self):
        # Environment configuration
        self.environment = os.environ.get('ENVIRONMENT', 'render')
        self.config = self.load_config()
        
        # Single optimized model for faster deployment
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Optimized vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        self.is_trained = False
        self.training_in_progress = False
        self.feedback_buffer = []
        self.performance_metrics = {}
        
        # Sample training data for quick deployment
        self.sample_data = self.get_sample_training_data()
        
    def load_config(self):
        """Load environment-specific configuration"""
        if self.environment == 'render':
            return {
                'max_features': 2000,
                'use_sample_data': True,
                'quick_train': True
            }
        else:
            return {
                'max_features': 5000,
                'use_sample_data': False,
                'quick_train': False
            }
    
    def get_sample_training_data(self):
        """Get sample training data for quick deployment"""
        # Sample hate speech and non-hate speech examples
        hate_examples = [
            "I hate all people from that country",
            "Those people are disgusting and should be eliminated",
            "Kill all terrorists and their families",
            "That group of people are subhuman",
            "I wish harm upon those protesters",
            "All immigrants should be deported immediately",
            "That religion is evil and dangerous",
            "Violence against that community is justified",
            "Those people don't deserve basic rights",
            "I hope something bad happens to them"
        ]
        
        normal_examples = [
            "I love spending time with my family",
            "The weather is beautiful today",
            "I'm looking forward to the weekend",
            "This movie was really entertaining",
            "I enjoy reading books in my free time",
            "The food at that restaurant was delicious",
            "I'm grateful for my friends and colleagues",
            "Learning new skills is always rewarding",
            "Exercise helps me stay healthy and happy",
            "I appreciate diverse perspectives and cultures",
            "Technology has made our lives easier",
            "I disagree with that policy decision",
            "The traffic is really bad today",
            "I'm frustrated with this situation",
            "That was a challenging problem to solve"
        ]
        
        # Create balanced dataset
        texts = hate_examples + normal_examples
        labels = [1] * len(hate_examples) + [0] * len(normal_examples)
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })
    
    @monitor_performance
    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        if pd.isna(text):
            return ""
        
        try:
            # Convert to string and normalize
            text = str(text).lower()
            text = unicodedata.normalize('NFKD', text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', ' ', text)
            
            # Handle repeated characters
            text = re.sub(r'(.)\1{2,}', r'\1', text)
            
            # Remove extra punctuation and whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return ""
    
    @monitor_performance
    def train_model(self):
        """Train the model with available data"""
        try:
            self.training_in_progress = True
            logger.info("Starting optimized model training...")
            
            # Use sample data for quick deployment
            df = self.sample_data.copy()
            
            # Preprocess texts
            logger.info("Preprocessing texts...")
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            
            # Prepare features
            X = df['processed_text'].values
            y = df['label'].values
            
            # Fit vectorizer and transform
            logger.info("Fitting vectorizer...")
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'training_samples': len(df),
                'model_type': 'LogisticRegression',
                'training_time': datetime.now().isoformat()
            }
            
            logger.info(f"Model training completed! Accuracy: {accuracy:.4f}")
            
            self.is_trained = True
            self.save_model()
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return {"error": str(e)}
        finally:
            self.training_in_progress = False
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open('hate_speech_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def load_model(self):
        """Load a previously saved model"""
        try:
            with open('hate_speech_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            self.is_trained = True
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.info("No saved model found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """Predict hate speech"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        if self.training_in_progress:
            return {"error": "Model training in progress, please wait"}
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {"error": "Text is empty after preprocessing"}
            
            # Vectorize
            features = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            result = {
                "text": text,
                "prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech",
                "confidence": float(max(probability)),
                "hate_probability": float(probability[1]) if len(probability) > 1 else 0.0,
                "model_version": "optimized_v1"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_batch(self, texts):
        """Batch prediction"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            return {"predictions": results}
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {"error": f"Batch prediction error: {str(e)}"}
    
    def get_status(self):
        """Get model status"""
        return {
            "trained": self.is_trained,
            "training_in_progress": self.training_in_progress,
            "status": "ready" if self.is_trained else ("training" if self.training_in_progress else "not trained"),
            "model_version": "optimized_v1",
            "performance_metrics": self.performance_metrics,
            "config": self.config
        }

# Initialize detector
detector = OptimizedHateSpeechDetector()

# Flask Routes
@app.route('/')
def index():
    """API documentation page"""
    base_url = request.url_root.rstrip('/')
    return render_template_string(HTML_TEMPLATE, base_url=base_url)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "model_version": "optimized_v1",
        "environment": detector.environment
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model"""
    if detector.training_in_progress:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        })
    
    def train_async():
        try:
            logger.info("Starting async training...")
            metrics = detector.train_model()
            
            if 'error' not in metrics:
                logger.info("Training completed successfully")
            else:
                logger.error(f"Training failed: {metrics['error']}")
                
        except Exception as e:
            logger.error(f"Async training error: {str(e)}")
    
    # Start training in background
    thread = Thread(target=train_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": "Model training started. Check /status for progress."
    })

@app.route('/predict', methods=['POST'])
def predict_hate_speech():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        result = detector.predict(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        if len(texts) > 20:
            return jsonify({"error": "Too many texts (max 20)"}), 400
        
        result = detector.predict_batch(texts)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def model_status():
    """Model status endpoint"""
    return jsonify(detector.get_status())

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Auto-train model on startup
def auto_train_on_startup():
    """Automatically train model on startup"""
    time.sleep(5)  # Wait for app to start
    if not detector.is_trained and not detector.training_in_progress:
        try:
            logger.info("Auto-training model on startup...")
            detector.train_model()
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")

if __name__ == '__main__':
    # Try to load existing model first
    if not detector.load_model():
        logger.info("No existing model found. Starting training...")
        # Start auto-training in background
        thread = Thread(target=auto_train_on_startup)
        thread.daemon = True
        thread.start()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )