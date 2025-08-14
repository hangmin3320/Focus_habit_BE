import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.services.data_service import FocusDataset
from app.models.lstm_action_classifier import LSTMClassifier

# --- 하이퍼파라미터 설정 ---
INPUT_DIM = 10  # Feature 개수
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 2  # 출력 클래스 개수 (집중, 비집중)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 30
DATA_PATH = '../data'
MODEL_SAVE_PATH = '../app/models/checkpoints/'


def train():
    """ 모델 학습 전체 과정을 수행하는 함수 """

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    dataset = FocusDataset(data_dir=DATA_PATH, sequence_length=30)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting training.")
        return

    # data split
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data loaded. Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    model = LSTMClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).squeeze()

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).squeeze()
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataset)
        val_accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.pth'))
            print(f"   -> New best model saved with validation loss: {val_loss:.4f}")

    print("\nFinished Training.")
    print(f"Best model saved at {os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.pth')}")


if __name__ == '__main__':
    train()
