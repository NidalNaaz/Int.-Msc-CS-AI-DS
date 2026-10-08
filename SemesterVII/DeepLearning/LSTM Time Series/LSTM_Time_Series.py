import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 1. Generate Synthetic Data
np.random.seed(42)
time = np.arange(0, 730)
temperature = 15 + 10 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 2, size=len(time))

# 2. Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
temp_normalized = scaler.fit_transform(temperature.reshape(-1, 1)).flatten()

print(f"Dataset Size: {len(temp_normalized)} days")
print(f"Original Temp Range: {temperature.min():.1f}°C to {temperature.max():.1f}°C")

# 3. Create Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len].unsqueeze(-1)
        y = self.data[idx + self.seq_len]
        return x, y

# Hyperparameters
SEQ_LEN = 30
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8

train_size = int(len(temp_normalized) * TRAIN_SPLIT)
train_data = temp_normalized[:train_size]
test_data = temp_normalized[train_size - SEQ_LEN:]

train_dataset = TimeSeriesDataset(train_data, SEQ_LEN)
test_dataset = TimeSeriesDataset(test_data, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 4. Model Definition
class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_dim=32, num_layers=1):
        super(TemperatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        prediction = self.fc(last_step_out)
        return prediction.squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemperatureLSTM(input_size=1, hidden_dim=32, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 5. Training Loop
EPOCHS = 100
print("\n--- Training Model ---")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0 or epoch == 1:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] - MSE Loss: {avg_loss:.6f}")

# 6. Evaluation & Plotting
print("\n--- Evaluating Test Set & Plotting ---")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        predictions.append(pred.item())
        actuals.append(y_batch.item())

predictions_deg = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals_deg = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 5))
plt.plot(actuals_deg, label="Actual Temperature (°C)", color="blue", linewidth=2)
plt.plot(predictions_deg, label="Predicted Temperature (°C)", color="orange", linestyle="--", linewidth=2)
plt.title("Time-Series Temperature Prediction: Actual vs Predicted")
plt.xlabel("Days (Test Set)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.savefig("temperature_prediction.png")
print("Plot saved as 'temperature_prediction.png'")
plt.show()
