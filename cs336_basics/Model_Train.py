import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer
from Tranformer_Class import TransformerLM
from transformers import get_scheduler
import numpy as np

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


# model = TransformerLM(
#     vocab_size=32000,
#     d_model=512,
#     n_heads=8,
#    # d_ff=int(512 * 8 / 3),
#     num_layers=6,
#     max_seq_len=512
# )

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




import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

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


def train_impl(model, x, batch_size, context_length, device, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4)
    model.to(device)
    model.train()

   
    steps_per_epoch = len(x) // (batch_size * context_length)
    total_steps = epochs * steps_per_epoch
    #scheduler for effinctient training
    scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0

        batch_per_epoch = 100
        pbar = tqdm(range(batch_per_epoch), desc=f"Epoch {epoch+1}")

        


        for _ in pbar:
            input_ids, targets = dataloader(x, batch_size, context_length, device)
            logits = model.forward(input_ids)  # shape: (B, T, vocab_size)
            # For each of the B sequences,

            # For each of the S (context_length) tokens in the sequence,

            # The model outputs a vector of 32000 (vocab_size) scores—one per possible token it could predict next.

            #loss per batch
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if epoch%5 == 0:
                avg_loss = total_loss / batch_per_epoch
                loss_history.append(avg_loss)
                save_checkpoint(model, optimizer, epoch + 1, avg_loss, path=f"checkpoint_epoch{epoch+1}.pth")

            pbar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })


        print(f"Epoch {epoch+1} avg loss: {total_loss / 100:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {current_lr:.6f}")

    torch.save(model.state_dict(), "final_model.pth")

    return loss_history


#After training, your model can map:

#input_ids (batch_size, seq_len) → logits (batch_size, seq_len, vocab_size)
#Each slice of logits corresponds to a probability distribution predicting the next token for each position in the sequence.
def generate(model, prompt, max_new_tokens, tokenizer):
    #simple implementation
    # logits = model(input_ids)           # shape: (1, T, vocab_size)
    # next_token_logits = logits[:, -1, :]  # grab logits for the final token
    # probs = torch.softmax(next_token_logits, dim=-1)  # get probabilities
    # next_token = torch.multinomial(probs, num_samples=1)

    #loop it for generation
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # shape: (1, T)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0])


if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    #to train script
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
    # model = TransformerLM(
    #     vocab_size=32000,
    #     d_model=512,
    #     n_heads=8,
    # # d_ff=int(512 * 8 / 3),
    #     num_layers=6,
    #     max_seq_len=512
    # )


    # print("start train")
    # loss_history = train_impl(model, token_ids, batch_size=32, context_length=128, device='cpu', epochs=10)


    # import matplotlib.pyplot as plt

    # plt.plot(loss_history, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.show()