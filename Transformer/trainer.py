from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchinfo
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm # Progress bar

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


DATASET_DIR = Path('~/workspace/SignLanguage')

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

with (DATASET_DIR / 'landmarks.json').open() as f:
    landmarks = json.load(f)

signs_df = pd.read_csv(DATASET_DIR / 'train.csv')

train_df = signs_df.sample(frac=0.9)
val_df = signs_df.drop(train_df.index).sample(frac=1)

class ASLDataset(Dataset):
    def __init__(self, dataset_df, prepare):
        files = np.load(DATASET_DIR / 'data.npz')
        self.items = [torch.Tensor(files[str(i)]).to(device) for i in tqdm(dataset_df.sequence_id, desc='Loading data', total=len(dataset_df))]
        self.labels = torch.Tensor(dataset_df.label.values).long().to(device)
        self.prepare = prepare
    
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, index):
        return self.prepare(self.items[index]).float(), self.labels[index]

POINTS = torch.cat([torch.tensor(value).unfold(0,3,1) for value in landmarks.values()])
INDICES = np.load(DATASET_DIR / 'indices.npy')

def prepare(x):
    # Find "angles"
    view = x[:,POINTS]
    vectors = torch.stack(
        (
            view[...,1,:] - view[...,0,:],
            view[...,2,:] - view[...,1,:]
        ),
        dim=-2
    ).float()
    angles = torch.div(
        vectors.prod(dim=-2).sum(dim=-1),
        vectors.square().sum(dim=-1).sqrt().prod(dim=-1)
    )#.acos()

    # Coordinate normalisation
    coord_counts = (~x.isnan()).sum(dim=(0,1))
    coord_no_nan = x.clone()
    coord_no_nan[coord_no_nan.isnan()] = 0
    coord_mean = coord_no_nan.sum(dim=(0,1)) / coord_counts
    normed = x - coord_mean

    # Coords + Angles
    tensor = torch.cat((
        normed.flatten(-2),
        angles),1)

    tensor[tensor.isnan()] = 0

    return tensor

train_dataset = ASLDataset(train_df, prepare)
train_preload = train_dataset.items
val_dataset = ASLDataset(val_df, prepare)
val_preload = val_dataset.items
len(train_dataset), len(val_dataset)

def pad_batch(batch):
    max_frames = max([len(entry) for entry in batch])
    size = (max_frames, len(batch), len(batch[0][0]))
    padded = torch.zeros(size).to(device)
    mask = torch.full((len(batch), max_frames), True).to(device)
    for index, entry in enumerate(batch):
        frames = len(entry)
        padded[:frames, index] = entry
        mask[index, :frames] = False
        
    return padded, mask

def collate(batch):
    transposed = list(zip(*batch))
    sequence = list(transposed[0])
    X = [torch.Tensor(x).nan_to_num(nan=0).flatten(1).float().to(device) for x in sequence]
    src, mask = pad_batch(X)
    y = torch.Tensor(transposed[1]).long().to(device)
    return src, mask, y

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).to(device)

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.in_tokens = 534
        self.d_model = 800
        self.d_embed_ff = 1200
        self.out_tokens = 250
        self.nhead = 8
        self.d_ff = 800
        self.nlayers = 1
        self.dropout = 0.4
        
        self.embed = nn.Sequential(
            nn.Linear(self.in_tokens, self.d_embed_ff),
            nn.LayerNorm(self.d_embed_ff),
            nn.ReLU(),
            nn.Linear(self.d_embed_ff, self.d_model)
        )
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_ff, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)
        
        self.decoder = nn.Linear(self.d_model, self.out_tokens)

    def forward(self, src, mask) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = torch.cat([torch.zeros((1,src.size(1),self.d_model)).to(src), src],0)
        mask = torch.cat([torch.full((src.size(1), 1), False).to(mask), mask],1)
        
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = output[0]
        output = self.decoder(output)
        
        return output
    
model = TransformerModel().to(device)

learning_rate = 5e-4
weight_decay = 1e-1
cycle = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle, eta_min=learning_rate / 10)

def train_val_loop(epoch, train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, n_offset=1):
    total_batches = len(train_dataloader)
    train_size, train_batches = 0, 0
    train_loss, train_correct = 0, 0
    train_preds, train_labels = [], []
    val_size, val_batches = 0, 0
    val_loss, val_correct = 0, 0
    val_preds, val_labels = [], []
    
    with tqdm(desc=f'Epoch {epoch+n_offset}', total=total_batches) as bar:
        
        # Training
        for batch, (src, mask, y) in enumerate(train_dataloader):
            
            # Compute prediction and loss
            pred = model(src, mask)
            loss = loss_fn(pred, y)
            
            # Compute metrics
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_preds.extend(pred.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            train_size += len(y)
            train_batches += 1

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
                
            scheduler.step(epoch + batch / total_batches)
            
            # Update progress bar
            bar.update()
            bar.set_postfix(accuracy = train_correct / train_size, loss = train_loss / train_batches,
                           lr=scheduler.get_last_lr())
            #bar.set_postfix(accuracy=train_correct / train_size, loss=train_loss / train_batches)

            
        bar.set_postfix(accuracy = train_correct / train_size, loss = train_loss / train_batches)
        #scheduler.step()
           
        # Validation
        with torch.no_grad():

            for batch, (src, mask, y) in enumerate(val_dataloader):
                
                # Compute prediction and loss
                pred = model(src, mask)
                val_loss += loss_fn(pred, y).item()
                
                # Compute metrics
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                val_preds.extend(pred.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                val_size += len(y)
                val_batches += 1

                # Update progress bar
                bar.set_postfix(
                    accuracy = train_correct / train_size, loss = train_loss / train_batches,
                    val_accuracy = val_correct / val_size, val_loss = val_loss / val_batches
                )

        # Compute metrics
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_precision = precision_score(train_labels, train_preds, average='weighted')
        train_recall = recall_score(train_labels, train_preds, average='weighted')

        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch + n_offset}:")
        print(f"Train - Loss: {train_loss / train_batches}, Accuracy: {train_correct / train_size}, F1: {train_f1}, Precision: {train_precision}, Recall: {train_recall}")
        print(f"Val   - Loss: {val_loss / val_batches}, Accuracy: {val_correct / val_size}, F1: {val_f1}, Precision: {val_precision}, Recall: {val_recall}")


    return train_correct / train_size, train_loss / train_batches, val_correct / val_size, val_loss / val_batches, val_f1, val_precision, val_recall

best_loss = float('inf')
#best_loss = 1.13 at 1000
saved_state = model.state_dict()

epochs = 200

# Iterate through epochs for training
for epoch in range(epochs):
    acc, loss, v_acc, v_loss, v_f1, v_precision, v_recall  = train_val_loop(epoch, train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)
    if v_loss<best_loss:
        best_loss = v_loss
        saved_state = model.state_dict()

torch.save(saved_state, 'model_weights_200.pth')
