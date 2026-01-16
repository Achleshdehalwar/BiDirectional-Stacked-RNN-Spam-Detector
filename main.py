import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import optuna
import json
import os

# --- CONFIG ---
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"System initialized. Running on: {DEVICE}")

# --- 1. DATA & UTILS ---
def get_data():
    print(">>> Downloading data...")
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    df = pd.read_csv(f"{path}/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

class SimpleTokenizer:
    def __init__(self, texts=None, max_vocab=None, vocab_dict=None):
        if vocab_dict:
            self.vocab = vocab_dict
        else:
            word_counts = Counter()
            for text in texts:
                tokens = re.findall(r'\w+', text.lower())
                word_counts.update(tokens)
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            for word, _ in word_counts.most_common(max_vocab - 2):
                self.vocab[word] = len(self.vocab)

    def encode(self, text, max_len):
        tokens = re.findall(r'\w+', text.lower())
        indices = [self.vocab.get(t, 1) for t in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

class SpamDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        encoded = self.tokenizer.encode(text, self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --- 2. MODEL DEFINITION ---
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        hidden_forward = hidden[-2,:,:]
        hidden_backward = hidden[-1,:,:]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        return self.sigmoid(self.fc(final_hidden))

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # A. Prepare Data
    df = get_data()
    tokenizer = SimpleTokenizer(texts=df['text'].values, max_vocab=MAX_VOCAB_SIZE)
    
    # Save the vocab NOW so we don't lose it
    print(">>> Saving vocabulary...")
    with open('vocab.json', 'w') as f:
        json.dump(tokenizer.vocab, f)

    train_size = int(0.8 * len(df))
    train_ds = SpamDataset(df[:train_size], tokenizer, MAX_SEQ_LEN)
    test_ds = SpamDataset(df[train_size:], tokenizer, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # B. Optuna Tuning
    print(">>> Starting Optuna Search...")
    def objective(trial):
        emb_dim = trial.suggest_int("embedding_dim", 32, 128)
        hid_dim = trial.suggest_int("hidden_dim", 32, 128)
        n_layers = trial.suggest_int("n_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        model = TextRNN(len(tokenizer.vocab), emb_dim, hid_dim, n_layers, dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for epoch in range(3): # Fast search
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x).squeeze(1), y)
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = (model(x).squeeze(1) > 0.5).float()
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            trial.report(correct/total, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return correct/total

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # 10 trials is enough for demo
    
    best = study.best_params
    print(f">>> Best Params: {best}")

    # C. Train Final Model & Save
    print(">>> Training Final Model...")
    final_model = TextRNN(len(tokenizer.vocab), best['embedding_dim'], best['hidden_dim'], 
                          best['n_layers'], best['dropout']).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=best['lr'])
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        final_model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(final_model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Complete")

    # D. Save Everything
    print(">>> Saving Model and Config...")
    torch.save(final_model.state_dict(), 'spam_rnn.pth')
    
    # We need to save the config so the App knows how to build the model shape
    config = {
        'vocab_size': len(tokenizer.vocab),
        'embedding_dim': best['embedding_dim'],
        'hidden_dim': best['hidden_dim'],
        'n_layers': best['n_layers'],
        'dropout': best['dropout']
    }
    with open('config.json', 'w') as f:
        json.dump(config, f)
        
    print("DONE! Files ready for upload: spam_rnn.pth, vocab.json, config.json")