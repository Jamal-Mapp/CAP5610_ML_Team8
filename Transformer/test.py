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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

# get test data
with (DATASET_DIR / 'test_landmarks.json').open() as f:
    landmarks = json.load(f)

test_df = pd.read_csv(DATASET_DIR / 'test.csv')


class ASLTestDataset(Dataset):
    def __init__(self, dataset_df, prepare):
        files = np.load(DATASET_DIR / 'test_data.npz')
        self.items = [torch.Tensor(files[str(i)]).to(device) for i in tqdm( dataset_df.sequence_id,desc='Loading data', total=len(dataset_df))]
        self.labels = torch.Tensor(dataset_df.label.values).long().to(device)
        self.prepare = prepare
    
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, index):
        return self.prepare(self.items[index]).float(), self.labels[index]

POINTS = torch.cat([torch.tensor(value).unfold(0,3,1) for value in landmarks.values()])
INDICES = np.load(DATASET_DIR / 'test_indices.npy')

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
    
#model = TransformerModel().to(device)

test_dataset = ASLTestDataset(test_df, prepare)
test_preload = test_dataset.items
len(test_dataset)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate)

model = TransformerModel().to(device)

weights_path = "model_weights.pth"  # Path to the saved weights
model.load_state_dict(torch.load(weights_path, map_location=device))

model = model.to(device)

def get_confusion_matrix(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data, mask, labels in test_dataloader:
            data, mask, labels = data.to(device), mask.to(device), labels.to(device)
            outputs = model(data, mask)
            
            # Assume output logits are in `outputs.logits`
            logits = outputs
            
            # Get predictions
            _, preds = torch.max(logits, dim=1)  # Get the index of the max logit (predicted class)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    return cm

conf_matrix = get_confusion_matrix(model, test_dataloader, device)
