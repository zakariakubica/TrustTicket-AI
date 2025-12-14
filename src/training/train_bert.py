"""
BERT Text Classifier Training Script
TrustTicket - AI-Powered Ticket Scam Detection

This script fine-tunes DistilBERT on ticket listing descriptions to classify scams vs legitimate listings.
Optimized for RTX 3070 GPU with 8GB VRAM.

Author: Zakaria Kubica
Date: December 2025
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm

# Configuration
class Config:
    # Paths
    DATA_PATH = "data/raw/ticket_listings_dataset.csv"
    MODEL_SAVE_DIR = "models/bert"
    CHECKPOINT_DIR = "models/checkpoints"
    
    # Model parameters
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256  # Maximum token length for descriptions
    BATCH_SIZE = 16   # Optimal for RTX 3070 (8GB VRAM)
    NUM_EPOCHS = 10   # Small dataset, more epochs needed
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    
    # Training settings
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TicketListingDataset(Dataset):
    """PyTorch Dataset for ticket listings"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(config):
    """Load dataset and prepare train/val/test splits"""
    
    print("📊 Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Create text features by combining multiple fields
    df['text'] = (
        df['event_name'].fillna('') + ' ' +
        df['artist'].fillna('') + ' ' +
        df['venue'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        'Price: £' + df['price'].astype(str) + ' ' +
        'Platform: ' + df['platform'].fillna('') + ' ' +
        'Red flags: ' + df['red_flags'].fillna('none')
    )
    
    # Convert labels to binary (0=legitimate, 1=scam)
    label_map = {'legitimate': 0, 'scam': 1}
    df['label_encoded'] = df['label'].map(label_map)
    
    # Split data: 70% train, 10% val, 20% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].values,
        df['label_encoded'].values,
        test_size=(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_SEED,
        stratify=df['label_encoded'].values
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_SEED,
        stratify=temp_labels
    )
    
    print(f"\n✅ Data split:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val:   {len(val_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def create_data_loaders(train_texts, val_texts, test_texts, 
                        train_labels, val_labels, test_labels, 
                        tokenizer, config):
    """Create PyTorch DataLoaders"""
    
    train_dataset = TicketListingDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = TicketListingDataset(val_texts, val_labels, tokenizer, config.MAX_LENGTH)
    test_dataset = TicketListingDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    """Evaluate model on validation/test set"""
    
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, predictions, true_labels

def save_model(model, tokenizer, config, metrics):
    """Save model, tokenizer, and training metadata"""
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(config.MODEL_SAVE_DIR)
    tokenizer.save_pretrained(config.MODEL_SAVE_DIR)
    
    # Save training metadata
    metadata = {
        'model_name': config.MODEL_NAME,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_epochs': config.NUM_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'max_length': config.MAX_LENGTH,
        'test_accuracy': metrics['test_accuracy'],
        'test_precision': metrics['test_precision'],
        'test_recall': metrics['test_recall'],
        'test_f1': metrics['test_f1'],
        'device': str(config.DEVICE)
    }
    
    with open(os.path.join(config.MODEL_SAVE_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {config.MODEL_SAVE_DIR}")

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("🎯 TrustTicket BERT Classifier Training")
    print("="*60)
    
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    # Check GPU
    print(f"\n⚡ Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load and prepare data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_prepare_data(config)
    
    # Initialize tokenizer and model
    print(f"\n🤖 Loading {config.MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2
    )
    model.to(config.DEVICE)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts, val_texts, test_texts,
        train_labels, val_labels, test_labels,
        tokenizer, config
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs...")
    best_val_accuracy = 0
    training_history = []
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, config.DEVICE)
        
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
    
    # Load best model for final evaluation
    print(f"\n{'='*60}")
    print("📈 Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, config.DEVICE
    )
    
    print(f"\n🎯 Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n📊 Confusion Matrix:")
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    # Save final model
    metrics = {
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }
    save_model(model, tokenizer, config, metrics)
    
    # Save training history
    with open(os.path.join(config.MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {config.MODEL_SAVE_DIR}")
    print(f"Target accuracy: 87% | Achieved: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
