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
    def __init__(self, vocab_size, embeddings_size, heads, padding_idx, number_of_layers):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.vocab_size = vocab_size
        self.embedder = Embedder(vocab_size, embeddings_size, padding_idx)
        self.pe = PositionalEncoder(embeddings_size, max_seq_len=embeddings_size, dropout=0.1)
        self.decoder = Decoder(embeddings_size, heads=heads)
        self.layers = get_clones(DecoderLayer(embeddings_size, heads, 0.1), number_of_layers)
        self.toprobs = nn.Linear(embeddings_size, vocab_size)

    
    def forward(self, seq, mask=None):
        x = self.embedder(seq)
        x = self.pe(x)
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

    #largest_seq = 0
    #for col in data.columns:
        #obs_ = np.delete(obs, np.argwhere(obs == "<PAD>"))
        #tags = np.append(obs_[:-1], np.array(["<EOT>"]))
        #seq = [s for s in obs_[-1]]
        #conc = np.append(tags, seq)
        #conc = np.append(conc, ["<EOS>"])
        #if len(conc) > largest_seq:
        #    largest_seq = len(conc)
        #data_.append(conc)

    #for i, obs in enumerate(data_):
        #pad_length = largest_seq-obs.shape[0]
        #pad_array = ["<PAD>"]*pad_length
        #data_[i] = np.append(obs, pad_array)

    #data = np.array(data_)

    #vocab = set()
    #for obs in data:
    #    vocab.update(obs)

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
    for col in data.columns[8:-1]:
        elements.update(data[col].unique())

    elem_to_col = {}
    for i, element in enumerate(elements):
        data[element] = "<PAD>"

    for row in data.iterrows():
        values = np.unique(row[1][data.columns[8:-1-len(elements)]].values)
        values = np.delete(values, np.argwhere(values=="<PAD>"))
        for element in elements:
            if element in values:
                row[1][element] = element

    data.drop(data.columns[8:-1-len(elements)], axis=1, inplace=True)
    
    seq = data.pop("Sequence")
    data["Sequence"] = seq

    data.to_csv("dataset_processed_1.csv", index=False)
    

    return data, vocab, max_seq

def process_data(data, vocab, max_seq):
    token_to_id, id_to_token = {}, {}
    
    token_to_id["<PAD>"] = 0
    id_to_token[0] = "<PAD>"

    token_to_id["<EOS>"] = 1
    id_to_token[1] = "<EOS>"

    #token_to_id["<EOT>"] = 2
    #id_to_token[2] = "<EOT>"

    #token_to_id["<DUMMY>"] = 2
    #id_to_token[2] = "<DUMMY>"

    for i, token in enumerate(vocab):
        id = len(token_to_id.keys())
        if token != "<PAD>" and token != "<EOS>" and token != "<EOT>":
            token_to_id[token] = id
            id_to_token[id] = token
            id += 1

    seq = []
    #for obs in data:
    #    encoded_obs = [token_to_id[token] for token in obs]
    #    seq.append(encoded_obs)
    
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

def gen(data, max_length):
    tags_end_position = 95 # hardcoded for now
    test_Seq_no = 8
    seqpred=[]
    aa="<PAD>"
    for i in range(tags_end_position, max_length+tags_end_position-1):
        tags = data.loc[test_Seq_no].tolist()[0:tags_end_position]
        if len(seqpred)==0:
            tags.append(aa)
            c=1
        else:
            for a in range(len(seqpred)):
                tags.append(seqpred[a])
            c=0
        for a in range(max_length-len(seqpred)-c):
            tags.append('<PAD>')
        #print(c, len(tags), tags)
        mask=[]
        for tag in tags:
            #print (tag)
            if tag == "<PAD>" or tag == "<DUMMY>":
                mask.append(False)
            else:
                mask.append(True)
        for a in range(len(tags)):
            pass#print(tags[a],mask[a])
        sample = torch.from_numpy(np.array(make_tokens_from_tags(tags,token_to_id)))
        mask = torch.from_numpy(np.array(mask))
        tags = sample.type(torch.LongTensor)
        #mask = mask.type(torch.LongTensor)
        prediction = model(tags).transpose(1,2)
        print(prediction.shape)
        next_char_idx = sample_categorical(prediction[0, :, tags.shape[0] - 1]) #0.5
        if next_char_idx <= 2:
            # query += "*"
            pass#break
        print(int(next_char_idx), id_to_token[int(next_char_idx)])
        #query += str(chr(max(32, next_char_idx)))
        #prediction = prediction.view(-1, prediction.size(-1))
        #idx=int(torch.argmax(prediction[i+1]))
        #aa = id_to_token[idx]
        #seqpred.append(aa)
    print(seqpred)

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
        #print(_)
        query_ = torch.zeros(max_len).to(torch.long)
        query_[:len(query)] = query
        #print(make_sequence_from_tokens(query_, id_to_token))
        output = model(query_.unsqueeze(0).to(device))
        #print(output)
        next_char_idx = sample_categorical(output[0, :, len(query) - 1], 0) #0.5
        #print(next_char_idx)
        query = query.tolist()
        query.append(int(next_char_idx))
        query = torch.from_numpy(np.array(query))
        if int(next_char_idx) < 2:
            break
        #print(make_sequence_from_tokens(query, id_to_token))
        #print(query.shape)
    return query

def make_tokens_from_tags(tags, token_to_id):
    tokens = map(lambda x: token_to_id[x], tags)
    return (list(tokens))

def make_sequence_from_tokens(ids, id_to_token):
    sequence = map(lambda x: id_to_token[x], ids.tolist())
    return "".join(list(sequence))



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, vocab, max_seq = get_data(max_length=300)
    seq, token_to_id, id_to_token = process_data(data, vocab, max_seq)
    seq = torch.from_numpy(seq).to(device)

    x = seq
    y = torch.hstack((x[:,1:], torch.zeros(x.shape[0], 1, dtype=torch.int32))).to(device)
    
    seq_input = seq
    mask = []
    for i, s in enumerate(tqdm(seq_input, desc="Creating masks")):
        mask.append(create_mask(s, token_to_id["<PAD>"]))
    
    mask = torch.from_numpy(np.array(mask)).to(device)
    
    # embedding_size needs to be divisible by heads
    model = ProGen(vocab_size=len(vocab), embeddings_size=512, heads=8, padding_idx=token_to_id["<PAD>"], number_of_layers=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[60, 100, 150], gamma=0.1)

    batch_size = 16
    for epoch in range(200):
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

        if epoch % 1 == 0:
            print(f"EPOCH {epoch} TRAIN_LOSS {round(train_total_loss,3)} \
                    TRAIN_PERPLEXITY {round(math.exp(train_total_loss),3)} \
                    TEST_LOSS {round(test_total_loss,3)} \
                    TEST_PERPLEXITY {round(math.exp(test_total_loss),3)}")
            
        if epoch % 10 == 0 and epoch != 0:
            batch_idx = random.sample(range(seq.shape[0]), 1)    
            query = []
            for token in x[batch_idx][0]:
                query.append(token)
            query = torch.from_numpy(np.array(query))[0:168+11]
            #gen = generate(model, query, token_to_id["<PAD>"])
            #print(make_sequence_from_tokens(gen, id_to_token))
            sampled  = make_sequence_from_tokens(sample_sentence(model, query,
                                                          max_len = 290,
                                                          temperature = 0), id_to_token)
            print(sampled)

