"""
BERT Inference Script
TrustTicket - AI-Powered Ticket Scam Detection

Test the trained BERT model on new ticket listings.

Author: Zakaria Kubica
Date: December 2025
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os

class ScamDetector:
    """BERT-based scam detection inference"""
    
    def __init__(self, model_dir="models/bert"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Loading model from {model_dir}...")
        print(f"⚡ Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Training accuracy: {self.metadata['test_accuracy']:.4f}")
        print(f"   F1 Score: {self.metadata['test_f1']:.4f}")
    
    def predict(self, text, return_probs=True):
        """
        Predict if a ticket listing is a scam
        
        Args:
            text: Listing description/details
            return_probs: If True, return probabilities for both classes
        
        Returns:
            dict with prediction and probabilities
        """
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        prediction = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        label = "SCAM" if prediction == 1 else "LEGITIMATE"
        confidence = float(probs[prediction])
        
        result = {
            'prediction': label,
            'confidence': confidence,
            'scam_probability': float(probs[1]),
            'legitimate_probability': float(probs[0])
        }
        
        return result

def test_examples():
    """Test the model on example listings"""
    
    detector = ScamDetector()
    
    print("\n" + "="*60)
    print("🧪 Testing BERT Scam Detector")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Legitimate Listing',
            'text': 'Taylor Swift Eras Tour Wembley Stadium June 2025 Price: £150 Platform: StubHub Verified resale'
        },
        {
            'name': 'Suspicious Low Price',
            'text': 'Taylor Swift Eras Tour tickets URGENT SALE ONLY £20!!! Contact me directly on WhatsApp Red flags: extremely_low_price,urgency_language'
        },
        {
            'name': 'Verified Platform',
            'text': 'Ed Sheeran Mathematics Tour Manchester Arena Price: £85 Platform: Ticketmaster Official resale Red flags: none'
        },
        {
            'name': 'Crypto Payment Request',
            'text': 'Coldplay tickets cheap price must sell today accept crypto only no refunds Red flags: crypto_payment,urgency_language,no_verification'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'-'*60}")
        print(f"Test {i}: {case['name']}")
        print(f"{'-'*60}")
        print(f"Input: {case['text'][:80]}...")
        
        result = detector.predict(case['text'])
        
        print(f"\n🎯 Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"   Scam Probability: {result['scam_probability']*100:.2f}%")
        print(f"   Legitimate Probability: {result['legitimate_probability']*100:.2f}%")
        
        # Risk assessment
        scam_prob = result['scam_probability']
        if scam_prob > 0.8:
            risk = "🔴 HIGH RISK"
        elif scam_prob > 0.5:
            risk = "🟡 MEDIUM RISK"
        else:
            risk = "🟢 LOW RISK"
        
        print(f"   Risk Level: {risk}")

if __name__ == "__main__":
    test_examples()
