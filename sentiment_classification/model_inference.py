import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

## ---------specify the device here --------
device = torch.device("cuda")
print(f"device: {device}")

## --------specify the model and data paths here --------
model_path = "FinBERT_best_acu.pth"
model_path_dir = "./model"  # should be the path to the FinBERT model, a folder containing config.json, pytorch_model.bin, vocab.txt
data_path = "data.xlsx" 

# hyperparameters
max_len = 512
batch_size = 64
hidden_dim = 64
num_classes = 3

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

tokenizer = BertTokenizer.from_pretrained(model_path_dir)
bert_model = BertModel.from_pretrained(model_path_dir).to(device)
model = FinBERT_BiLSTM_Attention(bert_model, hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

data = pd.read_excel(data_path)
texts = list(data["Contents"])
test_dataset = EmotionDatasetForPrediction(texts, tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_probs = []
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits, _ = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)  # 计算每个类别的概率
        predicted_labels = torch.argmax(probs, dim=1)
        all_preds.extend(predicted_labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# make predictions and probabilities into DataFrame
all_probs = pd.DataFrame(all_probs, columns=['Negative_Prob', 'Neutral_Prob', 'Positive_Prob'])
data = pd.concat([data, all_probs], axis=1)
all_preds = pd.DataFrame(all_preds, columns=['Predicted_Label'])
data = pd.concat([data, all_probs, all_preds], axis=1)
data.to_excel(data_path, index=False)
print(f"Test predictions with probabilities saved to {data_path}")