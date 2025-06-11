

import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import re
import warnings
warnings.filterwarnings("ignore")

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
        # Method 1: Direct paraphrasing
        input_text = f"paraphrase: {text}"
        
        # Tokenize
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Generate multiple variations
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
        
        # Decode results
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(decoded)
        
        return results
    
    def humanize_with_flan_t5(self, text):
        """Humanize using FLAN-T5 with instruction prompting"""
        # Create instruction prompt
        prompt = f"""Rewrite this text to sound more human, casual, and conversational. Use contractions, informal language, and a friendly tone:

Text: {text}

Rewritten:"""
        
        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate
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
        
        # Decode
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the result
            if "Rewritten:" in decoded:
                decoded = decoded.split("Rewritten:")[-1].strip()
            results.append(decoded)
        
        return results
    
    def humanize_with_gpt2(self, text):
        """Humanize using GPT-2 with prompt engineering"""
        # Create a prompt that encourages casual rewriting
        prompt = f"""Formal: {text}
Casual: """
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=2,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode('\n')[0]  # Stop at newline
            )
        
        # Decode and extract the casual version
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            if "Casual: " in decoded:
                casual_part = decoded.split("Casual: ")[-1].strip()
                # Remove any additional text after newline
                casual_part = casual_part.split('\n')[0]
                results.append(casual_part)
        
        return results
    
    def humanize_with_bart(self, text):
        """Use BART to create more concise, human-like versions"""
        # BART is primarily for summarization, but can make text more concise
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
            
            # Return the best result (first one for now)
            if results and results[0].strip():
                best_result = results[0]
                print(f"✨ Humanized: {best_result}")
                return best_result, results  # Return best + all options
            else:
                print("⚠️ Model returned empty result, returning original")
                return text, [text]
                
        except Exception as e:
            print(f"❌ Error during humanization: {e}")
            return text, [text]
    
    def compare_models(self, text):
        """Compare results from different models"""
        models = ["t5_paraphraser", "flan_t5", "gpt2"]
        results = {}
        
        for model in models:
            try:
                print(f"\n--- Testing {model} ---")
                temp_humanizer = AIHumanizer(model)
                result, _ = temp_humanizer.humanize_text(text)
                results[model] = result
            except Exception as e:
                print(f"Failed to test {model}: {e}")
                results[model] = "Failed to load"
        
        return results

# Demo and Testing
if __name__ == "__main__":
    print("🚀 AI HUMANIZER - Pre-trained Model Version")
    print("=" * 60)
    
    # Ask user which model to use
    print("Available models:")
    print("1. t5_paraphraser - Best for paraphrasing")
    print("2. flan_t5 - Good for instruction following")
    print("3. gpt2 - Good for creative rewriting")
    print("4. bart - Good for concise versions")
    
    choice = input("\nChoose model (1-4) or press Enter for default: ").strip()
    
    model_map = {
        "1": "t5_paraphraser",
        "2": "flan_t5", 
        "3": "gpt2",
        "4": "bart",
        "": "t5_paraphraser"  # default
    }
    
    selected_model = model_map.get(choice, "t5_paraphraser")
    
    # Initialize humanizer with selected model
    humanizer = AIHumanizer(selected_model)
    
    # Test cases
    test_texts = [
        "I am writing to inform you that the implementation of the aforementioned solution is optimal for achieving the desired outcome.",
        
        "Please be advised that we will process your request in a timely manner. Should you have any additional questions, please do not hesitate to contact us.",
        
        "The research indicates that the methodology employed is effective. Furthermore, the results demonstrate significant improvement."
    ]
    
    print(f"\n--- DEMO WITH {selected_model.upper()} ---")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 TEST CASE {i}")
        print("-" * 40)
        best_result, all_results = humanizer.humanize_text(text)
        
        print(f"\n📝 ORIGINAL: {text}")
        print(f"🎯 BEST RESULT: {best_result}")
        
        if len(all_results) > 1:
            print(f"\n🔄 ALL VARIATIONS:")
            for j, result in enumerate(all_results, 1):
                print(f"   {j}. {result}")
        
        print("-" * 40)
    
    # Interactive mode
    print(f"\n🎮 INTERACTIVE MODE - Using {selected_model}")
    print("=" * 60)
    
    while True:
        user_input = input("\n📝 Enter AI text to humanize (or 'quit'): ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using AI Humanizer!")
            break
        
        if user_input.strip():
            best_result, all_results = humanizer.humanize_text(user_input)
            
            print(f"\n🎯 HUMANIZED: {best_result}")
            
            if len(all_results) > 1:
                show_all = input("\nSee all variations? (y/n): ").lower() == 'y'
                if show_all:
                    for j, result in enumerate(all_results, 1):
                        print(f"   {j}. {result}")
            
            # Copy to clipboard option
            try:
                import pyperclip
                copy_choice = input("\nCopy result to clipboard? (y/n): ").lower()
                if copy_choice == 'y':
                    pyperclip.copy(best_result)
                    print("✅ Copied to clipboard!")
            except ImportError:
                print("💡 Install pyperclip for clipboard functionality: pip install pyperclip")