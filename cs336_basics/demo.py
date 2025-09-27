from BPETokenizer import BPETokenizer
import regex as re
from collections import defaultdict
import json
import base64

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer
from Tranformer_Class import TransformerLM
from transformers import get_scheduler
import numpy as np

from generate import generate



tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")

bad_lines = []

with open("data/tiny_stories_ids.jsonl", "r") as f:
    
    count = 0
    for i, line in enumerate(f, start=1):

        if count == 50:
            break
        
        try:
            obj = json.loads(line)
            print(obj)
            print(tokenizer.decode(obj))
            count = count + 1
        except json.JSONDecodeError as e:
            print(f"Error on line {i}: {e}")
            bad_lines.append((i, line.strip()))


#intitialize model architecture, should match the model you will load
model = TransformerLM(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
    # d_ff=int(512 * 8 / 3),
        num_layers=6,
        max_seq_len=512
    )  # Recreate the model with the same architecture


print('\n')
###########################################################
print("GENERATE DEMO \n")
print("you can see the function adds its generatd token to the end of the new input, so the text grows")
#load model
model.load_state_dict(torch.load("data/final_model_epoch_6.pth"))
#load tokenizer
tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")
#generate
print(generate(model,"In the garden, the squirrel climbed up the", 10, tokenizer))