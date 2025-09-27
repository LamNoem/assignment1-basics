# torch, nn, F: PyTorch core modules for building and training models.

# BPETokenizer: Your custom tokenizer for encoding/decoding text.

# TransformerLM: Your Transformer-based language model.

# get_scheduler: From HuggingFace Transformers, used to create a learning rate scheduler.

# numpy: For numerical operations.
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer
from Tranformer_Class import TransformerLM
from transformers import get_scheduler
import numpy as np


# model = TransformerLM(
#     vocab_size=32000,
#     d_model=512,
#     n_heads=8,
#    # d_ff=int(512 * 8 / 3),
#     num_layers=6,
#     max_seq_len=512
# )

import ast
import torch
import numpy as np
import random
#Loads token sequences from a .jsonl file and prepares batches for training.
def dataloader_from_jsonl(
    jsonl_path: str,
    batch_size: int,
    context_length: int,
    device: str
):
    """
    Loads token sequences from a .jsonl file where each line is a list of token IDs.
    Samples input-target pairs of fixed length.

    Args:
        jsonl_path (str): Path to the .jsonl file
        batch_size (int): Number of sequences per batch
        context_length (int): Length of each input sequence
        device (str): PyTorch device string ('cpu' or 'cuda:0')

    Returns:
        input_tensor (torch.Tensor): shape (B, context_length)
        target_tensor (torch.Tensor): shape (B, context_length)
    """
    sequences = []

    # Step 1: Read and parse each line as a Python list
    # Reads each line as a Python list of token IDs.

    # Filters out malformed or too-short sequences.
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                tokens = ast.literal_eval(line.strip())
                if isinstance(tokens, list) and len(tokens) >= context_length + 1:
                    sequences.append(tokens)
            except (ValueError, SyntaxError):
                continue  # skip malformed lines

    if len(sequences) == 0:
        raise ValueError("No valid sequences found with sufficient length.")

    # Step 2: Sample random subsequences
    input_batch = []
    target_batch = []

    # Randomly selects a sequence and extracts a window of context_length tokens.

    # input_seq: tokens from t to t+context_length

    # target_seq: tokens from t+1 to t+1+context_length (next-token prediction)
    for _ in range(batch_size):
        
        seq = random.choice(sequences)
        max_start = len(seq) - context_length - 1
        start = np.random.randint(0, max_start + 1)
        input_seq = seq[start : start + context_length]
        target_seq = seq[start + 1 : start + 1 + context_length]
        input_batch.append(input_seq)
        target_batch.append(target_seq)

    # Step 3: Convert to tensors
    #Converts batches to PyTorch tensors and moves them to the specified device.
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)

    return input_tensor, target_tensor




import torch
import os
#Saves model and optimizer state to disk.
def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
#Loads model and optimizer state from disk.
#never ended up using this
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# After loading, call model.train() or model.eval() depending on your use case.

# If resuming training, continue from epoch + 1.

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm

#Trains the Transformer model using the JSONL token data.
def train_impl(model, data_file_Path, batch_size, context_length, device, epochs):
    # Uses cross-entropy loss for next-token prediction.

    # AdamW optimizer with learning rate 5e-4.

    # Moves model to device and sets to training mode.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4)
    model.to(device)
    model.train()

    #Linear learning rate scheduler with warmup (5% of total steps).
    batch_per_epoch = 100
    total_steps = epochs * batch_per_epoch
    # implementscheduler for efficient training
    #adapts the learning rate based on the dynamics of the training process
    scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=(5/100)*total_steps,
    num_training_steps=total_steps
    )
    
    #training_loop
    loss_history = []
    for epoch in range(epochs):

        # Tracks loss per epoch.

        # Uses tqdm for progress bar.
        total_loss = 0
        batch_per_epoch = 100
        pbar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch+1}")

        


        for _ in pbar:
            #FORWARD PASS
            #Loads a batch of input-target pairs.

            #Gets model predictions (logits).
            input_ids, targets = dataloader_from_jsonl(data_file_Path, batch_size, context_length, device)
            logits = model.forward(input_ids)  # shape: (B, S, vocab_size)
            # For each of the B sequences,

            # For each of the S (context_length) tokens in the sequence,

            # The model outputs a vector of 32000 (vocab_size) scores 
            # one per possible token it could predict next.

            #loss per batch
            #Reshapes logits and targets to match expected shape for CrossEntropyLoss.
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            #BACKWARD PASS

            #Standard training steps: zero gradients, backpropagate, update weights, adjust learning rate.
            #did not implement this from scratch, used libraries
            #zero the optomizer gradients for this batch.
            optimizer.zero_grad()
            #backpropogation, calculating the loss gradient according to each parameter
            loss.backward()
            #update model parameters according to loss gradient
            optimizer.step()
            #update scheduler
            scheduler.step()

            #Updates progress bar with current loss and learning rate.
            total_loss += loss.item()
            # if epoch%5 == 0:
            #     avg_loss = total_loss / batch_per_epoch
            #     loss_history.append(avg_loss)
            #     save_checkpoint(model, optimizer, epoch + 1, avg_loss, path=f"checkpoint_epoch_2{epoch+1}.pth")
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

        #Logs average loss and learning rate.
        avg_loss = total_loss / batch_per_epoch
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1} avg loss: {total_loss / batch_per_epoch:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {current_lr:.6f}")

    #Saves final model weights and returns loss history.
    #torch.save(model.state_dict(), "final_model.pth")
    torch.save(model.state_dict(), f"data/final_model_epoch_2{epochs}.pth")

    return loss_history




#main script
if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    #to train script
    #loads bpe tokenizer from files
    tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")


    
    #token_ids = np.array(token_list, dtype=np.uint16)

    import numpy as np
    import json

    #np.save("tiny_stories_ids_np.npy", token_ids)
    # Save as JSON
    #with open("tiny_stories_ids.json", "w") as f:
        # token_list = tokens.tolist()
        #json.dump(token_list, f)


 #-------------------------------------------------------------------------------------
 #Creates Transformer model with specified architecture.
    model = TransformerLM(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
    # d_ff=int(512 * 8 / 3),
        num_layers=6,
        max_seq_len=512
    )

    #Trains model on tokenized TinyStories dataset.
    print("start train")
    loss_history = train_impl(model, "data/tiny_stories_ids.jsonl", batch_size=32, context_length=128, device='cpu', epochs=6)

    #plot loss curve
    import matplotlib.pyplot as plt

    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()