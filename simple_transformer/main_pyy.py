import torch, sys
import pandas as pd
import numpy as np  
from torch import nn
import torch.nn.functional as F
from model import Transformer
import torch.distributions as dist

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
    
    for _ in range(max_len - len(query)):
        #print(_)
        query_ = torch.zeros(max_len).to(torch.long)
        query_[:len(query)] = query
        #print(make_sequence_from_tokens(query_, id_to_token))
        output, _     = model(query_.unsqueeze(0).to(device))
        #print(output)
        next_char_idx = sample_categorical(output[0, :, len(query) - 1], 0) #0.5
        #print(next_char_idx)

        query = query.tolist()
        query.append(int(next_char_idx))
        query = torch.from_numpy(np.array(query))
        #print(make_sequence_from_tokens(query, id_to_token))
        #print(query.shape)

    
    return query


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

def get_data(max_length):
    data = pd.read_csv("dataset.csv")
    data = data.replace(np.nan, '<DUMMY>', regex=True)
    #data.drop("Unnamed: 0", axis=1, inplace=True)
    #data.drop("Entry", axis=1, inplace=True)
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

def make_sequence_from_tokens(ids, id_to_token):
    sequence = map(lambda x: id_to_token[x], ids.tolist())
    return " ".join(list(sequence))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 50

learning_rate, batch_size, epochs = 1e-2, 16, 200

data, vocab, max_seq = get_data(max_length=100)
seq, token_to_id, id_to_token = process_data(data, vocab, max_seq)

token_end=81

max_len= seq.shape[1]

# Preprocess strings into tensors of char ascii indexes
inputs  = torch.zeros(seq.shape).to(torch.long).to(device)
targets = torch.zeros(seq.shape).to(torch.long).to(device)

for i, tag in enumerate(seq):
     #print(i,tag)
     inputs[i,0:len(seq[i])]   = torch.from_numpy(seq[i])
     #print(i,inputs[i])
     targets[i,token_end:len(seq[i])+token_end] = torch.from_numpy(seq[i][token_end:])
     #print(i,targets[i])
# Split into train and test dataset

combined = torch.stack([inputs, targets], dim=1)
train_size = int(0.8 * len(combined))
test_size = len(combined) - train_size
train_ds, test_ds = torch.utils.data.random_split(combined, [train_size, test_size])

train_x, train_y = combined[train_ds.indices][:, 0, :], combined[train_ds.indices][:, 1, :]
test_x, test_y   = combined[test_ds.indices][:, 0, :],  combined[test_ds.indices][:, 1, :]



max_index = int(max(train_x.max(), test_x.max()))

args = {
    'emb_dim':        32,            # Embedding vector dimension
    'n_att_heads':    16,            # Number of attention heads for each transformer block
    'n_transformers': 4,             # Depth of the network (nr. of self-attention layers)
    'seq_length':     max_len,       # Sequence length
    'num_tokens':     max_index + 1, # Vocabulary size (highest index found in dataset)
    'device':         device,        # Device: cuda/cpu
    'wide':           False          # Narrow or wide self-attention
}

stats = { 'train_loss': [], 'test_loss': [], 'perplexity': [] } # we accomulate and save training statistics here
model = Transformer(**args).to(device)
opt   = torch.optim.Adam(lr=learning_rate, params=model.parameters())

for i in range(epochs):
    model.train()
    opt.zero_grad()
    
    # Sample a random batch of size `batch_size` from the train dataset
    idxs = torch.randint(size=(batch_size,), low=0, high=len(train_x))
    
    output, (emb_mean, emb_max) = model(train_x[idxs])
    loss = F.nll_loss(output, train_y[idxs], reduction='mean')
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    loss.backward()
    opt.step()
    
    # Calculate perplexity on the test-set
    model.eval()
    output_test, _ = model(test_x)
    loss_on_test   = F.nll_loss(output_test, test_y, reduction='mean')
    perplexity     = torch.exp(loss_on_test).item()

    # Update the stats and print something.
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(loss_on_test.item())
    stats['perplexity'].append(perplexity)
    
    sampled  = sample_sentence(model, test_y[1,0:token_end],
                               max_len = max_len,
                               temperature = 0.5)[token_end+1:]
    
    protein= []
    
    for aa in sampled:
        amino_acid = id_to_token[int(aa)]
        protein.append(amino_acid)
    protein=" ".join(list(protein))
        
    
    
    
    to_print = [
        f"EPOCH %03d"        % i,
        f"LOSS %4.4f"        % stats['train_loss'][-1],
        f"PERPLEXITY %4.4f" % stats['perplexity'][-1],
        f"\t%s"      % protein
    ]
    print(" ".join(to_print))

# Finally, save everyting:
torch.save({
    'state_dict':   model.state_dict(), 
    'stats':        stats,
    'args':         args,
    'train_x':      train_x,
    'test_x':       test_x
}, f"words.model.pth")