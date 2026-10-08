import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. DATASET & PREPROCESSING
# ==========================================
text = "machine learning is awesome! character level lstm model in pytorch."

# Build Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char2idx[ch] for ch in s if ch in char2idx]

def decode(indices):
    return "".join([idx2char[i] for i in indices])

class CharDataset(Dataset):
    def __init__(self, text, seq_len, char2idx):
        self.seq_len = seq_len
        self.data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

# Hyperparameters
SEQ_LEN = 8
BATCH_SIZE = 4
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
EPOCHS = 200
LEARNING_RATE = 0.01

dataset = CharDataset(text, seq_len=SEQ_LEN, char2idx=char2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. LSTM MODEL ARCHITECTURE
# ==========================================
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]
        out = self.embedding(x)                     # [batch_size, seq_len, embed_dim]
        out, hidden = self.lstm(out, hidden)         # out: [batch_size, seq_len, hidden_dim]
        logits = self.fc(out)                       # [batch_size, seq_len, vocab_size]
        return logits, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        
        # Flatten outputs for CrossEntropyLoss: [batch_size * seq_len, vocab_size] vs [batch_size * seq_len]
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if epoch % 40 == 0 or epoch == 1:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ==========================================
# 4. INFERENCE / GENERATION FUNCTION
# ==========================================
def predict_next_chars(model, seed_text, num_predict=10, temperature=1.0):
    model.eval()
    current_input = seed_text
    
    with torch.no_grad():
        for _ in range(num_predict):
            # Encode seed text
            encoded = encode(current_input[-SEQ_LEN:]) # Keep last SEQ_LEN chars
            if not encoded:
                break
            x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device) # [1, seq_len]
            
            # Forward pass
            logits, _ = model(x)
            
            # Extract prediction for the very LAST character in sequence
            last_char_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_char_logits, dim=-1)
            
            # Sample next character index
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]
            
            current_input += next_char
            
    return current_input

# Quick test run
test_seed = "mach"
generated = predict_next_chars(model, seed_text=test_seed, num_predict=12)
print(f"\n[Test Inference] Seed: '{test_seed}' -> Generated: '{generated}'")

# Save checkpoint
torch.save(model.state_dict(), "lstm_model.pth")
print("\nModel saved to 'lstm_model.pth'.")

# ==========================================
# 5. FASTAPI DEPLOYMENT ENDPOINT
# ==========================================
app = FastAPI(title="Char-Level LSTM API")

class GenerationRequest(BaseModel):
    seed: str
    num_chars: int = 10
    temperature: float = 1.0

@app.post("/predict")
def generate_text(req: GenerationRequest):
    result = predict_next_chars(
        model, 
        seed_text=req.seed, 
        num_predict=req.num_chars, 
        temperature=req.temperature
    )
    return {
        "seed": req.seed,
        "prediction": result,
        "next_character": result[len(req.seed):]
    }

# Run API server (Uncomment to launch server directly from script)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
