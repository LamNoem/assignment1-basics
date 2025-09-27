import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer
from Tranformer_Class import TransformerLM
from transformers import get_scheduler
import numpy as np

#After training, your model can map:

#input_ids (batch_size, seq_len) → logits (batch_size, seq_len, vocab_size)
#Each slice of logits corresponds to a probability distribution predicting the next token for each position in 
# the sequence.
def generate(model, prompt, max_new_tokens, tokenizer):
    #model: A pre-trained language model (e.g., GPT).
    #prompt: Initial text input to start generation.

    #max_new_tokens: Number of tokens to generate beyond the prompt.

    #tokenizer: Converts text to token IDs and vice versa."


    #simple implementation
    # logits = model(input_ids)           # shape: (1, T, vocab_size)
    # next_token_logits = logits[:, -1, :]  # grab logits for the final token
    # probs = torch.softmax(next_token_logits, dim=-1)  # get probabilities
    # next_token = torch.multinomial(probs, num_samples=1)
    #loop it for generation

    #set model to eval for inference
    #eval - is used to generate so dropout and other functions will not activate
    #no backpropogation
    model.eval()
    
    #encode the prompt
    input_ids = tokenizer.encode(prompt)
    #model takes tensors
    #(batch_size, sequence_length)
    input_ids = torch.tensor([input_ids])  # Batch dimension, model must take in this dimension


    #Moves the tensor to the same device (CPU/GPU) as the model.
    #i dont really understand this, but apparently the data has to be on the same device where the 
    #computations are being made
    input_ids = input_ids.to(next(model.parameters()).device)
    

    #Iterates to generate one token at a time until reaching max_new_tokens
    for _ in range(max_new_tokens):

        #Forward Pass
        #Gets raw predictions (logits) from the model for each token in the sequence.
        #Shape: (batch_size, sequence_length, vocab_size)
        logits = model(input_ids)

        
        temp = 1 #reduces randomness if below 1

        #Extracts logits for the last token in the sequence.
        #This is where the model predicts the next token.
        #tensor slicing operation to extract the logits (raw predictions) for the last token in the input sequence.

        #logits.shape = (batch_size, sequence_length, vocab_size)
        #batch_size: Number of sequences processed at once (usually 1 during generation).

        #sequence_length: Number of tokens in the input.

        #vocab_size: Number of possible tokens the model can predict.

        #Each position in the sequence has a vector of logits representing the model’s prediction for the next token.
        #think like a cube, where the 2D face (Batch, Sequence), and the depth is the prob to be the next token for each vocab item (vocab size)

        #the slicing:

        #Slice	Meaning
        #:	All batches (usually just one)
        #-1	The last token in the sequence
        #:	All logits (i.e., the full vocabulary)
        #So this line extracts the logits for the last token in each batch.
        next_token_logits = logits[:, -1, :]

        #Makes a copy of the current input sequence to apply repetition penalty.
        generated = input_ids.clone()
        # Apply repetition penalty
        #lowers score of what already is in prompt
        #Penalizes tokens that have already appeared in the sequence.

        #Reduces their logits to discourage repetition.
        #if a token id appeard in the generated text, its logit/probability will be reduced
        repetition_penalty = 1.5
        for token_id in set(generated[0].tolist()):
            next_token_logits[0, token_id] /= repetition_penalty


        #Applies softmax to convert logits to probabilities.

        #Samples one token based on the probability distribution.
        probs = torch.softmax(next_token_logits/temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        #append the token we generated
        input_ids = torch.cat([input_ids, next_token], dim=1)
        print(input_ids)


 


    #decode takes a list not tensor
    return tokenizer.decode(input_ids[0].tolist())

#intitialize model architecture, should match the model you will load
model = TransformerLM(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
    # d_ff=int(512 * 8 / 3),
        num_layers=6,
        max_seq_len=512
    )  # Recreate the model with the same architecture

#load model
model.load_state_dict(torch.load("data/final_model_epoch_6.pth"))
#load tokenizer
tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")
#generate
print(generate(model,"In the garden, the squirrel climbed up the", 10, tokenizer))