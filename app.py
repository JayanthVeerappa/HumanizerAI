from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import re
import warnings
import threading
import time
import os
warnings.filterwarnings("ignore")

# Initialize Flask app with static file serving
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for frontend connection

class AIHumanizer:
    def __init__(self, model_choice="t5_paraphraser"):
        """
        Initialize with different pre-trained models
        Options: 't5_paraphraser', 'flan_t5', 'gpt2', 'bart'
        """
        print(f"🤖 Loading AI Humanizer with {model_choice}...")
        self.model_choice = model_choice
        
        if model_choice == "t5_paraphraser":
            self.load_t5_paraphraser()
        elif model_choice == "flan_t5":
            self.load_flan_t5()
        elif model_choice == "gpt2":
            self.load_gpt2()
        elif model_choice == "bart":
            self.load_bart()
        else:
            raise ValueError("Invalid model choice")
    
    def load_t5_paraphraser(self):
        """Load T5 model fine-tuned specifically for paraphrasing"""
        print("📥 Loading T5 Paraphraser...")
        try:
            self.model_name = "ramsrigouthamg/t5_paraphraser"
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            print("✅ T5 Paraphraser loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load T5 Paraphraser: {e}")
            self.load_flan_t5()  # Fallback
    
    def load_flan_t5(self):
        """Load Google's FLAN-T5 for instruction following"""
        print("📥 Loading FLAN-T5...")
        try:
            self.model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print("✅ FLAN-T5 loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load FLAN-T5: {e}")
            raise
    
    def load_gpt2(self):
        """Load GPT-2 for text generation"""
        print("📥 Loading GPT-2...")
        try:
            self.model_name = "gpt2-medium"
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ GPT-2 loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load GPT-2: {e}")
            raise
    
    def load_bart(self):
        """Load BART for text summarization/paraphrasing"""
        print("📥 Loading BART...")
        try:
            self.pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                framework="pt"
            )
            print("✅ BART loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load BART: {e}")
            raise
    
    def humanize_with_t5(self, text):
        """Humanize using T5 paraphraser"""
        input_text = f"paraphrase: {text}"
        
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=200,
                num_return_sequences=3,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(decoded)
        
        return results
    
    def humanize_with_flan_t5(self, text):
        """Humanize using FLAN-T5 with instruction prompting"""
        prompt = f"""Rewrite this text to sound more human, casual, and conversational. Use contractions, informal language, and a friendly tone:

Text: {text}

Rewritten:"""
        
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=300,
                num_return_sequences=2,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            if "Rewritten:" in decoded:
                decoded = decoded.split("Rewritten:")[-1].strip()
            results.append(decoded)
        
        return results
    
    def humanize_with_gpt2(self, text):
        """Humanize using GPT-2 with prompt engineering"""
        prompt = f"""Formal: {text}
Casual: """
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=2,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode('\n')[0]
            )
        
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            if "Casual: " in decoded:
                casual_part = decoded.split("Casual: ")[-1].strip()
                casual_part = casual_part.split('\n')[0]
                results.append(casual_part)
        
        return results
    
    def humanize_with_bart(self, text):
        """Use BART to create more concise, human-like versions"""
        try:
            result = self.pipeline(text, max_length=150, min_length=30, do_sample=True)
            return [result[0]['summary_text']]
        except Exception as e:
            print(f"BART error: {e}")
            return [text]
    
    def humanize_text(self, text):
        """Main humanization function - routes to appropriate model"""
        print(f"\n🤖 Original: {text}")
        print(f"🔄 Using {self.model_choice} model...")
        
        try:
            if self.model_choice == "t5_paraphraser":
                results = self.humanize_with_t5(text)
            elif self.model_choice == "flan_t5":
                results = self.humanize_with_flan_t5(text)
            elif self.model_choice == "gpt2":
                results = self.humanize_with_gpt2(text)
            elif self.model_choice == "bart":
                results = self.humanize_with_bart(text)
            
            if results and results[0].strip():
                best_result = results[0]
                print(f"✨ Humanized: {best_result}")
                return best_result, results
            else:
                print("⚠️ Model returned empty result, returning original")
                return text, [text]
                
        except Exception as e:
            print(f"❌ Error during humanization: {e}")
            return text, [text]

# Global model instances - loaded once when server starts
humanizers = {}

def load_model(model_choice):
    """Load model in background thread"""
    if model_choice not in humanizers:
        try:
            humanizers[model_choice] = AIHumanizer(model_choice)
        except Exception as e:
            print(f"Failed to load {model_choice}: {e}")
            return False
    return True

# STATIC FILE ROUTES - ADD THESE
@app.route('/')
def serve_frontend():
    """Serve the main HTML file"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Update the API URL in the HTML for production
        if 'localhost' not in request.host:
            html_content = html_content.replace(
                "const API_BASE_URL = 'http://localhost:5000/api';",
                "const API_BASE_URL = '/api';"
            )
        
        return html_content
    except FileNotFoundError:
        return "index.html file not found", 404

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 404

@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve other static files"""
    try:
        return send_from_directory('.', filename)
    except FileNotFoundError:
        return f"File {filename} not found", 404

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(humanizers.keys()),
        'timestamp': time.time()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    models = {
        't5_paraphraser': 'T5 Paraphraser - Best Overall',
        'flan_t5': 'FLAN-T5 - Instruction Following', 
        'gpt2': 'GPT-2 - Creative Rewriting',
        'bart': 'BART - Concise Versions'
    }
    return jsonify({
        'models': models,
        'loaded': list(humanizers.keys())
    })

@app.route('/api/humanize', methods=['POST'])
def humanize_text():
    """Main humanization endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text'].strip()
        model_choice = data.get('model', 't5_paraphraser')
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 5000:  # Limit text length
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        # Load model if not already loaded
        if not load_model(model_choice):
            return jsonify({'error': f'Failed to load model: {model_choice}'}), 500
        
        # Humanize text
        humanizer = humanizers[model_choice]
        best_result, all_results = humanizer.humanize_text(text)
        
        return jsonify({
            'success': True,
            'original': text,
            'humanized': best_result,
            'variations': all_results,
            'model_used': model_choice,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare results from different models"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        models_to_compare = ['t5_paraphraser', 'flan_t5', 'gpt2', 'bart']
        results = {}
        
        for model in models_to_compare:
            try:
                if not load_model(model):
                    results[model] = {'error': 'Failed to load model'}
                    continue
                
                humanizer = humanizers[model]
                best_result, all_results = humanizer.humanize_text(text)
                
                results[model] = {
                    'success': True,
                    'humanized': best_result,
                    'variations': all_results
                }
                
            except Exception as e:
                results[model] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'original': text,
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Compare API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preload', methods=['POST'])
def preload_model():
    """Preload a specific model"""
    try:
        data = request.get_json()
        model_choice = data.get('model', 't5_paraphraser')
        
        # Load model in background thread
        def load_in_background():
            load_model(model_choice)
        
        thread = threading.Thread(target=load_in_background)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Loading {model_choice} in background',
            'model': model_choice
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Humanizer API Server...")
    print("=" * 60)
    
    # Preload default model
    print("📥 Preloading default model...")
    load_model('t5_paraphraser')
    
    print("✅ Server ready!")
    
    # Get port from environment variable (required for Render)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    if debug_mode:
        print("🌐 Access at: http://localhost:5000")
        print("📋 API Endpoints:")
        print("   - GET  /api/health     - Health check")
        print("   - GET  /api/models     - Available models")
        print("   - POST /api/humanize   - Humanize text")
        print("   - POST /api/compare    - Compare models")
        print("   - POST /api/preload    - Preload model")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)