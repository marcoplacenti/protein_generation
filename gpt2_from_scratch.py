import torch.nn.functional as F
import torch
import torch.nn as nn

import math
from torch.autograd import Variable
import numpy as np
import random
import pandas as pd
from tqdm.std import tqdm


class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size, max_seq_len=500, dropout = 0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, embedding_size)
        for pos in range(max_seq_len):
            for i in range(0, embedding_size):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_size)))
                else:
                    pe[pos, i] = math.cos(pos / (10000 ** ((2 * (i))/embedding_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        #add constant to embedding
        seq_len = x.size(1)
        self.pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        #if x.is_cuda:
            #self.pe.cuda()
        x = x + self.pe
        return self.dropout(x)
    

class Decoder(nn.Module):
    def __init__(self, embedding_size, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_size)
        self.norm_2 = Norm(embedding_size)
        
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.attn_1 = MultiHeadAttention(heads, embedding_size, dropout=0.1)
        self.ff = FeedForward(embedding_size, dropout=0.1)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Norm(nn.Module):
    def __init__(self, embedding_size, eps = 1e-6):
        super().__init__()
    
        self.size = embedding_size
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_size, dropout = 0.1):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.d_k = embedding_size // heads
        self.h = heads
        
        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * embedding_size
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.embedding_size)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, embedding_size, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(embedding_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_size)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
        

class ProGen(nn.Module):
    def __init__(self, vocab_size, embeddings_size, heads, padding_idx, number_of_layers):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedder = Embedder(vocab_size, embeddings_size, padding_idx)
        self.pe = PositionalEncoder(embeddings_size, max_seq_len=embeddings_size, dropout=0.1)
        self.decoder = Decoder(embeddings_size, heads=heads)
        #self.layers = get_clones(DecoderLayer(embeddings_size, 8, 0.1), number_of_layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, seq, mask):
        x = self.embedder(seq)
        x = self.pe(x)
        x = self.decoder(x, mask)
        return x


def create_mask(s, token_to_id, id_to_token):
    """dim = s.shape[0]
    s_mask = np.full((dim, dim), True)
    for i in range(dim):
        s_mask[i][i+1:] = False
    """
    pad_id = token_to_id["<PAD>"]
    dim = s.shape[0]
    s_mask = np.full((dim, dim), False)
    for i in range(dim):
        curr_seq = s[:i+1]
        for j, token in enumerate(curr_seq):
            if token != pad_id:
                s_mask[i][j] = True
    
    return s_mask


def get_data(max_length):
    data = pd.read_csv("dataset.csv")
    data = data.replace(np.nan, '<DUMMY>', regex=True)
    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.drop("Entry", axis=1, inplace=True)
    data = data[data["Sequence"].map(len) <= max_length]
    vocab = set()
    for col in data.columns:
        if col != "Sequence":
            vocab.update(data[col])

    seq_len = []
    max_seq = 0
    for seq in data["Sequence"]:
        seq = [s for s in seq]
        seq_len.append(len(seq))
        if len(seq) > max_seq:
            max_seq = len(seq)
        vocab.update(seq)

    vocab.update(["<PAD>"])
    vocab.update(["<EOS>"])

    return data, vocab, max_seq

def process_data(data, vocab, max_seq):
    token_to_id, id_to_token = {}, {}
    
    token_to_id["<PAD>"] = 0
    id_to_token[0] = "<PAD>"

    token_to_id["<EOS>"] = 1
    id_to_token[1] = "<EOS>"

    token_to_id["<DUMMY>"] = 2
    id_to_token[2] = "<DUMMY>"

    for i, token in enumerate(vocab):
        cum_i = len(token_to_id.keys())
        if token != "<PAD>" and token != "<EOS>" and token != "<DUMMY>":
            token_to_id[token] = cum_i
            id_to_token[cum_i] = token
            cum_i += 1

    seq = []
    for record in data.values:
        tags = record[:-1]
        sequence = record[-1]
        
        encoded_record = [token_to_id[tag] for tag in tags]

        for char in sequence:
            encoded_record.append(token_to_id[char])
        encoded_record.append(token_to_id["<EOS>"])
        
        if len(sequence) < max_seq:
            for i in range(max_seq-len(sequence)):
                encoded_record.append(token_to_id["<PAD>"])

        seq.append(encoded_record)

    return np.array(seq), token_to_id, id_to_token

if __name__ == "__main__":
    data, vocab, max_seq = get_data(max_length=40)
    seq, token_to_id, id_to_token = process_data(data, vocab, max_seq)
    seq = torch.from_numpy(seq)
    #seq = torch.from_numpy(np.random.randint(1, 100, size=(5000,15)))
    
    seq_input = seq[:, :-1]
    mask = []
    for i, s in enumerate(tqdm(seq_input, desc="Creating masks")):
        mask.append(create_mask(s, token_to_id, id_to_token))
    
    mask = torch.from_numpy(np.array(mask))

    #heads = 8
    #emb_size = int(len(vocab)/heads)+1
    #print(emb_size, emb_size*heads)
    
    # embedding_size needs to be divisible by heads
    model = ProGen(vocab_size=len(vocab), embeddings_size=len(vocab), heads=1, padding_idx=token_to_id["<PAD>"], number_of_layers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)

    batch_size = 64
    for epoch in range(20):
        total_loss = 0

        batch_idx = random.sample(range(seq.shape[0]), batch_size)
        samples_batched = seq[batch_idx]

        batch_input = samples_batched[:,:-1]
        sampled_masks = mask[batch_idx]
        ys = samples_batched[:, 1:].contiguous().view(-1)
        
        preds = model(batch_input, sampled_masks)
       
        optimizer.zero_grad()
               
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(epoch, total_loss)
    
