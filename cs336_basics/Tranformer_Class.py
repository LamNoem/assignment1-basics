# torch: Core PyTorch library.

# nn: Neural network components (layers, modules).

# F: Functional API for activation functions, etc.

# BPETokenizer: Your custom tokenizer for converting text to token IDs.
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer

#Defines a single Transformer block with pre-layer normalization.
#much of the math and computation within the transformer blocks, i do not understand.
class PreNormBlock(nn.Module):
    # we are defining functions used in architecture
    def __init__(self, d_model, n_heads):
        #d_model: Dimensionality of token embeddings.
        #n_heads: Number of attention heads.

        #Initializes the base nn.Module.
        super().__init__()

        #Defines hidden size for the feed-forward network (usually larger than d_model).
        d_ff = int(d_model * 8 / 3)

        #Layer normalization before attention.
        self.ln1 = nn.LayerNorm(d_model)
        #Multi-head self-attention layer. batch_first=True means input shape is (batch, seq, dim).
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        #Layer normalization before feed-forward network.
        self.ln2 = nn.LayerNorm(d_model)

        #Feed-forward network:

        # Projects up to d_ff

        # Applies ReLU

        # Projects back to d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Step 1: LayerNormalization before attention
        x_ln1 = self.ln1(x)
         # Step 2: Multi-head self-attention
         #ignore attn_weights by using _
        attn_output, _ = self.attn(x_ln1, x_ln1, x_ln1)
        # Step 3: Residual connection
        x_res1 = x + attn_output
        # Step 4: LayerNormalization before FFN
        x_ln2 = self.ln2(x_res1)

        # Step 5: Feed-forward network with ReLU
        ffn_output = self.ffn(x_ln2)

        # Step 6: Second residual connection + output
        return x_res1 + ffn_output
    


#overall architecture
class TransformerLM(nn.Module):
# vocab_size: Size of vocabulary.

# d_model: Embedding dimension.

# n_heads: Attention heads per block.

# num_layers: Number of Transformer blocks.

# max_seq_len: Maximum sequence length.
    #defining functions to be used in arhitecture
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, num_layers=6, max_seq_len=512):
        #Initializes base nn.Module.
        super().__init__()
        #Embed token IDs into d_model-dimensional vectors.
        self.token_embed = nn.Embedding(vocab_size, d_model)

#Creates a lookup table of shape (max_seq_len, d_model)
# Each position index (0 to max_seq_len - 1) maps to a learnable vector of size d_model
# These vectors are added to token embeddings to inject positional information into the model
#Why It’s Needed
# Transformers process all tokens in parallel and don’t inherently understand order. 
# Positional embeddings solve this by giving each token a sense of "where" it is in the sequence.
#You're adding a unique learned vector to each token based on its position in the sequence.
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        #Embeds positional indices to inject order information.
        #Transformers process tokens in parallel, so they need positional info to understand order.
        #d_model is the size of the vector that will describe the position of a token within a sentence.


        #creates a list of instances
        #Creates a list of PreNormBlock instances to form the Transformer stack.
        self.blocks = nn.ModuleList([
            PreNormBlock(d_model, n_heads) for _ in range(num_layers)
        ])

        #defines layer normalization after all blocks.
        self.final_norm = nn.LayerNorm(d_model)
        #Projects final hidden states to vocabulary logits for prediction.
        #complicated math shape -> proper shape for vocabulary predictions
        #So for each token position, it takes a vector of size d_model and transforms it into a vector of size vocab_size.
        self.output_proj = nn.Linear(d_model, vocab_size)

    #Takes input tensor of token IDs: shape (batch_size, seq_len).
    def forward(self, input_ids):
        #token embedding layer
        #output: embedded vector representations (batch_size, sequence_length, d_model)
        #for/in each sentence (batch size), each token id (sequence length), will have d_model size vector representation

        #Extracts batch size B and sequence length S.
        B, S = input_ids.shape
        #Creates positional indices for each token in the batch. creates shape for next step
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        #embed token ids
        #Adds token embeddings and positional embeddings.
        #x.shape -> (batch_size, seq_len, d_model), after this line
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        #Passes input through each Transformer block sequentially.
        for block in self.blocks:
            x = block(x)

        #Applies final layer normalization.
        x = self.final_norm(x)

        # Output Normalization and Embedding
        #take the final activations and turn them into a distribution over the vocabulary
        # Linear projection to vocabulary logits
        #Assuming x.shape = (batch_size, seq_len, d_model), this line applies the linear transformation to each token’s 
        # hidden state.
        #logits.shape = (batch_size, seq_len, vocab_size)
        #Each token in sequence now has a vector of logits representing the likelihood for every possible token 
        # in the vocabulary to be the next token.
        #imagine a cube, where the depth is a vector of probabilities, one for each vocab id.
        logits = self.output_proj(x)

        return logits
    



