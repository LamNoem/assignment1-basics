#Transformer Language Model Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer

#parameters
batch_size = 2
sequence_length = 5
vocab_size = 32000 #size of vocabulary
d_model = 512 #dimension of embedding vector / how many numbers in the vector used to represent one token id


#Dummy batch of token ids, improve and use tokenizer (batch_size, sequence_length)
#each integer is a token id, each sequence is a "sentence or sample of text of the same length"
#batch_size determines the number of sentences we are storing, passing on
#sequence_length, determines the length of those sentences

#Token Embeddings

tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")
lst = tokenizer.encode(" ")
print(lst)
string = tokenizer.decode(lst)
print(string)
print("######")

cot = 0
inputs = []
batches = 0
temp = []
for i in tokenizer.encode_iterable("data/story.txt"):
         if batches >= batch_size:
                 break
         cot +=1
         temp.append(i)
         print(i)
         print(tokenizer.decode([i]))
         if cot >= sequence_length:
                 inputs.append(temp)
                 temp = []
                 cot = 0
                 batches += 1
             
print(inputs)
inputs = torch.tensor(inputs)

#input_ids = torch.tensor([
 #   [12,45,378,201,7],
  #  [89,4,9987,13,44]
#], dtype = torch.long) #shape: (2,5)

#token embedding layer
embedding_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)

#output: embedded vector representations (batch_size, sequence_length, d_model)
#for/in each sentence (batch size), each token id (sequence length), will have d_model size vector representation
embedded = embedding_layer(inputs)

print("Input ids shape: ", inputs.shape)
print("Embedded shape:", embedded.shape)

#print (inputs)
#for i in inputs:
   #     for t in i:
      #          print (tokenizer.decode([int(t)]))

print (embedded)

x = embedded

def PreNorm_forward_TransformerBlock (embedded_input, d_model, n_heads):

        #Pre-norm Transformer Block
        # x: (batch_size, seq_len, d_model)
        #n_heads = 8 #factor of d_model
        d_ff = d_model * 8/3

        # Step 1: LayerNorm before attention
        layernorm1 = nn.LayerNorm(d_model)
        x_ln1 = layernorm1(x)

        # Step 2: Multi-head self-attention
        attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        #ignore attn_weights by using _
        attn_output, _ = attn(x_ln1, x_ln1, x_ln1)

        # Step 3: Residual connection
        x_res1 = x + attn_output

        # Step 4: LayerNorm before FFN
        layernorm2 = nn.LayerNorm(d_model)
        x_ln2 = layernorm2(x_res1)

        # Step 5: Feed-forward network with ReLU
        ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
                )
        ffn_output = ffn(x_ln2)

        # Step 6: Second residual connection
        x_out = x_res1 + ffn_output

        return x_out

#1
x = PreNorm_forward_TransformerBlock(x, d_model,8)
#2
x = PreNorm_forward_TransformerBlock(x, d_model,8)
#3
x = PreNorm_forward_TransformerBlock(x, d_model,8)
#4
x = PreNorm_forward_TransformerBlock(x, d_model,8)
#5
x = PreNorm_forward_TransformerBlock(x, d_model,8)
#6
x = PreNorm_forward_TransformerBlock(x, d_model,8)


#after last transformer block
final_layernorm = nn.LayerNorm(d_model)
final_output = final_layernorm(x)

print("Embedded shape:", x.shape)

# Output Normalization and Embedding
#take the final activations and turn them into a distribution over the vocabulary
# Linear projection to vocabulary logits
linear = nn.Linear(d_model, vocab_size)
logits = linear(x)  # Shape: (batch_size, seq_len, vocab_size)

# Softmax to convert logits to probabilities (during inference)
probs = F.softmax(logits, dim=-1)  # Still (batch_size, seq_len, vocab_size)

#Training a transformer LM
#define loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#define optimizer
from torch.optim import AdamW
