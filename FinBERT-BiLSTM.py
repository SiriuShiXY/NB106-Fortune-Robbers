import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


print("Finish loading libraries")

device = torch.device("cuda")
print(f"device: {device}")

model_path = r"/home/xs2337/FinBERT_L-12_H-768_A-12_pytorch/FinBERT_L-12_H-768_A-12_pytorch/"
data_path = r"/home/xs2337/artificially tagging data_seg.xlsx"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path).to(device)
print("Finish loading FinBERT model")

hidden_dim = 64
num_classes = 3  
test_size = 0.2
max_len = 512
batch_size = 32
epochs = 80


class FinBERT_BiLSTM(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_classes):
        super(FinBERT_BiLSTM, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.batch_norm(lstm_output[:, -1, :])
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)
        return logits

model = FinBERT_BiLSTM(bert_model, hidden_dim, num_classes).to(device)

data = pd.read_excel(data_path)
texts = list(data["title_seg"])
labels = list(data["emotion"])

labels = [label + 1 for label in labels] 

print(f"Original data size: {len(labels)}")

clean_texts = []
clean_labels = []
for text, label in zip(texts, labels):
    if pd.notnull(label) and label in [0.0, 1.0, 2.0]:  # 假设合法标签为0.0, 1.0, 2.0
        clean_texts.append(text)
        clean_labels.append(int(label))

print(f"Cleaned data size: {len(clean_labels)}")

X_train, X_valid, y_train, y_valid = train_test_split(clean_texts, clean_labels, test_size=test_size, random_state=42)
print(f"Train data size: {len(X_train)}")
print(f"Valid data size: {len(X_valid)}")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_len)
valid_dataset = EmotionDataset(X_valid, y_valid, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.lstm.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-4},
])

# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 40)

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        progress_bar.set_postfix(batch_loss=loss.item())

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(batch_loss=loss.item())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return total_loss / len(data_loader), accuracy, precision, recall, f1

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_precision, val_recall, val_f1 = eval_model(model, valid_loader, criterion, device)

    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))
    train_accuracies.append(float(train_acc))
    val_accuracies.append(float(val_acc))
    val_precisions.append(float(val_precision))
    val_recalls.append(float(val_recall))
    val_f1s.append(float(val_f1))

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}')

    # 保存模型权重
    # torch.save(model.state_dict(), f'FinBERT_epoch{epoch+1}_acc{val_acc:.4f}.pth')

    # 学习率调度
    scheduler.step()

val_loss, val_acc, val_precision, val_recall, val_f1 = eval_model(model, valid_loader, criterion, device)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}')

# epochs_range = range(1, epochs + 1)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, train_losses, label='Train Loss')
# plt.plot(epochs_range, val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Train and Validation Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
# plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Train and Validation Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, val_precisions, label='Validation Precision')
# plt.plot(epochs_range, val_recalls, label='Validation Recall')
# plt.xlabel('Epochs')
# plt.ylabel('Metrics')
# plt.title('Validation Precision and Recall')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, val_f1s, label='Validation F1 Score')
# plt.xlabel('Epochs')
# plt.ylabel('F1 Score')
# plt.title('Validation F1 Score')
# plt.legend()

# plt.tight_layout()
# plt.show()

