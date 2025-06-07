import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

print("Finish loading libraries")

## ---------specify the device here --------
device = torch.device("cuda")
print(f"device: {device}")

## --------specify the model and data paths here --------
model_path = "./model" # should be the path to the FinBERT model, a folder containing config.json, pytorch_model.bin, vocab.txt
data_path = "data.xlsx"

tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path).to(device)
print("Finish loading FinBERT model")

# hyperparameters
hidden_dim = 64
num_classes = 2
test_size = 0.2
max_len = 512
batch_size = 64
epochs = 80

# Define the FinBERT BiLSTM model
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

# Define the Attention mechanism       
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weight)
    
    def forward(self, lstm_output):
        attention_scores = torch.matmul(lstm_output, self.attention_weight).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

# Define the FinBERT BiLSTM model with Attention mechanism
class FinBERT_BiLSTM_Attention(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_classes):
        super(FinBERT_BiLSTM_Attention, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        context_vector, attention_weights = self.attention(lstm_output)
        context_vector = self.batch_norm(context_vector)
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)
        return logits, attention_weights

# Define the dataset class for training and validation
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
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EmotionDatasetForPrediction(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

model = FinBERT_BiLSTM_Attention(bert_model, hidden_dim, num_classes).to(device)

data = pd.read_excel(data_path)
texts = list(data["Comment"])
labels = list(data["Label"])

augmented_labels = []

# split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(texts, labels, test_size=test_size, random_state=42)
print(f"Train data size: {len(X_train)}")
print(f"Valid data size: {len(X_valid)}")

train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_len)
valid_dataset = EmotionDataset(X_valid, y_valid, tokenizer, max_len)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.lstm.parameters(), 'lr': 3e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.attention.parameters(), 'lr': 3e-4},
])

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40)

def train_Attention_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs, _ = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        progress_bar.set_postfix(batch_loss=loss.item())

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)
    
def eval_Attention_model(model, data_loader, criterion, device):
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

            outputs, _ = model(input_ids, attention_mask)
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

best_val_loss = float('inf')
best_val_acc = 0.0

for epoch in range(epochs):
    train_loss, train_acc = train_Attention_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_precision, val_recall, val_f1 = eval_Attention_model(model, valid_loader, criterion, device)
    
    if val_loss < best_val_loss and val_acc > best_val_acc:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'FinBERT_best_loss.pth')
        print(f'Best model saved at epoch {epoch + 1} based on validation loss')
    
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

    scheduler.step()

val_loss, val_acc, val_precision, val_recall, val_f1 = eval_Attention_model(model, valid_loader, criterion, device)
print(f'Final Validation loss: {val_loss}, Validation accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}')
