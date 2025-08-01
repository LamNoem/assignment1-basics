import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer


class PreNormBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_ff = int(d_model * 8 / 3)

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Step 1: LayerNorm before attention
        x_ln1 = self.ln1(x)
         # Step 2: Multi-head self-attention
         #ignore attn_weights by using _
        attn_output, _ = self.attn(x_ln1, x_ln1, x_ln1)
        # Step 3: Residual connection
        x_res1 = x + attn_output
        # Step 4: LayerNorm before FFN
        x_ln2 = self.ln2(x_res1)

        # Step 5: Feed-forward network with ReLU
        ffn_output = self.ffn(x_ln2)

        # Step 6: Second residual connection
        return x_res1 + ffn_output
    



class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, num_layers=6, max_seq_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
#Creates a lookup table of shape (max_seq_len, d_model)
# Each position index (0 to max_seq_len - 1) maps to a learnable vector of size d_model
# These vectors are added to token embeddings to inject positional information into the model
#Why It’s Needed
# Transformers process all tokens in parallel and don’t inherently understand order. 
# Positional embeddings solve this by giving each token a sense of "where" it is in the sequence.
#You're adding a unique learned vector to each token based on its position in the sequence.
        self.pos_embed = nn.Embedding(max_seq_len, d_model)


        #creates a list of instances
        self.blocks = nn.ModuleList([
            PreNormBlock(d_model, n_heads) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        #token embedding layer
        #output: embedded vector representations (batch_size, sequence_length, d_model)
        #for/in each sentence (batch size), each token id (sequence length), will have d_model size vector representation
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        #embedding token ids
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)


        x = self.final_norm(x)

        # Output Normalization and Embedding
        #take the final activations and turn them into a distribution over the vocabulary
        # Linear projection to vocabulary logits
        logits = self.output_proj(x)

        return logits
    
import numpy as np
import torch

def dataloader(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Samples a batch of input-target pairs from a long token sequence.

    Args:
        x (np.ndarray): 1D array of token IDs (length n)
        batch_size (int): number of sequences per batch (B)
        context_length (int): length of each input sequence (m)
        device (str): PyTorch device string ('cpu' or 'cuda:0')

    Returns:
        input_tensor (torch.Tensor): shape (B, m)
        target_tensor (torch.Tensor): shape (B, m)
    """
    n = len(x)
    max_start = n - context_length - 1  # ensure room for target shift
    starts = np.random.randint(0, max_start, size=batch_size)

    #stacking (batch_size # of) token id sequences into one tensor
    input_batch = np.stack([x[i : i + context_length] for i in starts])
    target_batch = np.stack([x[i + 1 : i + 1 + context_length] for i in starts])

    #train
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    #corresponding next tokens
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)

    return input_tensor, target_tensor


