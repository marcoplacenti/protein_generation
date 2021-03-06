import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.optim.lr_scheduler import MultiStepLR

import math
from torch.autograd import Variable
import numpy as np
import random
import pandas as pd
from tqdm.std import tqdm
import copy
import difflib


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

    def forward(self, x, seq_len):
        #x = x * math.sqrt(self.embedding_size)
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

    def forward(self, x, mask=None):
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

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

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
        
class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(embedding_size)
        self.norm_2 = Norm(embedding_size)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, embedding_size, dropout=0.1)
        self.ff = FeedForward(embedding_size, dropout=0.1)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ProGen(nn.Module):
    def __init__(self, vocab_size, tensor_length, embeddings_size, heads, padding_idx, number_of_layers):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.vocab_size = vocab_size
        self.embedder = Embedder(vocab_size, embeddings_size, padding_idx)
        self.pe = PositionalEncoder(embeddings_size, max_seq_len=tensor_length, dropout=0.1)
        self.layers = get_clones(DecoderLayer(embeddings_size, heads, 0.1), number_of_layers)
        self.toprobs = nn.Linear(embeddings_size, vocab_size)

    
    def forward(self, seq, mask=None):
        x = self.embedder(seq)
        x = self.pe(x, seq.shape[0])
        for i in range(self.number_of_layers):
            x = self.layers[i](x, mask)
        #x = self.decoder(x, mask)
        
        batch, seq_len, emb_size = x.shape

        x = x.view(batch * seq_len, emb_size)   # [ batch*seq_len,  emb_dim ]
        x = self.toprobs(x)                                 # [ batch*seq_len,  num_tokens ]
        x = x.view(batch, self.vocab_size, seq_len) # [ batch, num_tokens, seq_len ]

        return F.log_softmax(x, dim=1)


def create_mask(s, pad_id): 
    pad_idx = np.argwhere(s == pad_id)

    dim = s.shape[0]
    s_mask = np.full((dim, dim), True)
    for i in range(dim):
        s_mask[i][i+1:] = False
        curr_seq = s[:i+1]
        s_mask[i][pad_idx] = False

    return s_mask


def get_data(max_length):
    data = pd.read_csv("dataset.csv")
    data = data.replace(np.nan, '<PAD>', regex=True)
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

    #vocab.update(["<EOT>"])
    vocab.update(["<PAD>"])
    vocab.update(["<EOS>"])

    elements = set()
    for col in data.columns[1:-9]:
        elements.update(data[col].unique())

    elem_to_col = {}
    for i, element in enumerate(elements):
        data[element] = "<PAD>"

    for row in data.iterrows():
        values = np.unique(row[1][data.columns[1:-9-len(elements)]].values)
        values = np.delete(values, np.argwhere(values=="<PAD>"))
        for element in elements:
            if element in values:
                row[1][element] = element

    data.drop(data.columns[1:-9-len(elements)], axis=1, inplace=True)
    
    seq = data.pop("Sequence")
    data["Sequence"] = seq

    data.to_csv("dataset_processed.csv", index=False)
    

    return data, vocab, max_seq

def process_data(data, vocab, max_seq):
    token_to_id, id_to_token = {}, {}
    
    token_to_id["<PAD>"] = 0
    id_to_token[0] = "<PAD>"

    token_to_id["<EOS>"] = 1
    id_to_token[1] = "<EOS>"

    for i, token in enumerate(vocab):
        id = len(token_to_id.keys())
        if token != "<PAD>" and token != "<EOS>":
            token_to_id[token] = id
            id_to_token[id] = token
            id += 1

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

        """
        enc_rec_no_tags = []#[token_to_id["<PAD>"] for tag in tags]
        for char in sequence:
            enc_rec_no_tags.append(token_to_id[char])
        enc_rec_no_tags.append(token_to_id["<EOS>"])
        #if len(sequence) < max_seq:
        for _ in range(max_seq-len(sequence)):
            enc_rec_no_tags.append(token_to_id["<PAD>"])

        seq.append(enc_rec_no_tags)
        """

    return np.array(seq), token_to_id, id_to_token

def sample_categorical(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    return dist.Categorical(p).sample()

def sample_sentence(model, query, max_len = 140, temperature=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(max_len - query.shape[0]):
        query_ = torch.zeros(max_len).to(torch.long)
        query_[:len(query)] = query
        output = model(query_.unsqueeze(0).to(device))
        next_char_idx = sample_categorical(output[0, :, len(query) - 1], 0) #0.5
        query = query.tolist()
        query.append(int(next_char_idx))
        query = torch.from_numpy(np.array(query))
        if int(next_char_idx) < 2:
            break
    return query


def make_sequence_from_tokens(ids, id_to_token):
    sequence = map(lambda x: id_to_token[x], ids.tolist())
    return " ".join(list(sequence))



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, vocab, max_seq = get_data(max_length=300)
    seq, token_to_id, id_to_token = process_data(data, vocab, max_seq)
    seq = torch.from_numpy(seq).to(device)

    x = seq
    y = torch.hstack((x[:,1:], torch.zeros(x.shape[0], 1, dtype=torch.int32))).to(device)
    
    mask = []
    for i, s in enumerate(tqdm(x, desc="Creating masks")):
        mask.append(create_mask(s, token_to_id["<PAD>"]))
    
    mask = torch.from_numpy(np.array(mask)).to(device)
    
    embedding_sizes = [32, 64, 128, 512]
    heads = [1, 2, 4, 8]
    no_stacked_layers = [3, 4, 5, 6]

    metrics = open('metrics.csv', 'w')
    metrics.write("EMBEDDING_SIZE, HEADS, NUMBER OF LAYERS, EPOCH, TRAIN_LOSS, TRAIN_PERP, TEST_LOSS, TEST_PERP\n")
    generations = open("generations.csv", "w")
    generations.write("EMBEDDING_SIZE, HEADS, NUMBER_OF_LAYERS, EPOCH, AVG_SIM, SAMPLE\n")

    for i in range(len(heads)):
        print("-----------------------------------------")
        print(f"RUNNING CONFIGURATION {i+1}/{len(heads)}...")
        embedding_size = embedding_sizes[i]
        head = heads[i]
        number_of_layers = no_stacked_layers[i]

        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp

        # embedding_size needs to be divisible by heads
        model = ProGen(vocab_size=len(vocab), embeddings_size=embedding_size, tensor_length=seq.shape[0], heads=head, padding_idx=token_to_id["<PAD>"], number_of_layers=number_of_layers).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=[60, 100, 150], gamma=0.1)

        batch_size = 32
        for epoch in tqdm(range(100), desc='Running epochs...'):
            batch_idx = random.sample(range(seq.shape[0]), batch_size*2)
            train_idx = batch_idx[:batch_size]
            test_idx = batch_idx[batch_size:]

            model.train()
            optimizer.zero_grad()

            train_input = x[train_idx]
            train_masks = mask[train_idx]
            train_y = y[train_idx]

            preds = model(train_input, train_masks)

            loss = F.nll_loss(preds, train_y, reduction='mean', ignore_index=0)
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            train_total_loss = loss.item()
            

            loss.backward()
            optimizer.step()
            scheduler.step()

            model.eval()
            
            test_input = x[test_idx]
            test_masks = mask[test_idx]
            test_y = y[test_idx]

            preds = model(test_input, test_masks)
            loss = F.nll_loss(preds, test_y, reduction='mean', ignore_index=0)
            
            test_total_loss = loss.item()

            metrics.write("%s, %s, %s, %s, %s, %s, %s, %s\n" % \
                (embedding_size, head, number_of_layers, epoch, \
                    str(round(train_total_loss,3)), str(round(math.exp(train_total_loss),3)), \
                    str(round(test_total_loss,3)), str(round(math.exp(test_total_loss),3))))
        
            
            if epoch % 10 == 0:# and epoch != 0:
                avg_sim = 0
                print(make_sequence_from_tokens(x[27], id_to_token))
                cond = torch.from_numpy(np.array(x[27]))[:169+10]
                generated_seq = sample_sentence(model, cond, max_len = 291, temperature = 0)[169+10:]
                sampled = make_sequence_from_tokens(generated_seq, id_to_token)
                print(sampled)
            
                avg_sim += difflib.SequenceMatcher(None, generated_seq, torch.from_numpy(np.array(x[0]))[169+10:]).ratio()
                generations.write("%s, %s, %s, %s, %s, %s\n" % (embedding_size, head, number_of_layers, epoch, avg_sim, sampled))


        print("-----------------------------------------")
        print()
        
        metrics.close()
        generations.close()
