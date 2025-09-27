import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from BPETokenizer import BPETokenizer
from Tranformer_Class import TransformerLM
from transformers import get_scheduler
import numpy as np






import torch
import os


# After loading, call model.train() or model.eval() depending on your use case.

# If resuming training, continue from epoch + 1.

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm







#--------------------------------------------------
global_tokenizer = None
#dont really understand, probably needed this for when i tried parallel encoding
def init_tokenizer():
    global global_tokenizer
    from BPETokenizer import BPETokenizer
    global_tokenizer = BPETokenizer.from_files(
        "data/owt_32k_merges_dict.json",
        "data/owt_32k_vocab.json",
        "data/owt_32k_merges.json"
    )

#identify size of text for progress bar
def count_lines(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        #sum 1 for each line
        return sum(1 for _ in f)

#yields blocks of text
def read_blocks(file_path, block_size=50):
    
    with open(file_path, 'r', encoding="utf-8") as file:
        block = []
        for line in file:
            block.append(line)
            if len(block) >= block_size:
                print("Yielding block")
                yield block
                block = []
        if block:
            print("Yielding final block")
            yield block

#ecodes blocks of text
def encode_block(lines):
    from BPETokenizer import BPETokenizer
    tokenizer = BPETokenizer.from_files(
        "data/owt_32k_merges_dict.json",
        "data/owt_32k_vocab.json",
        "data/owt_32k_merges.json"
    )
    print("encoding block")
    return [tokenizer.encode(line) for line in lines]







if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    #to train script
    #load tokenizer
    tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")


    total_lines = count_lines("data/TinyStoriesV2-GPT4-train.txt")
    #50 is the block size
    estimated_blocks = total_lines // 50 + 1

    #read_block yields blocks of text

    with open("data/tiny_stories_ids_2.jsonl", "w") as f:
        #yield blocks
        for block in tqdm(read_blocks("data/TinyStoriesV2-GPT4-train.txt"), total=estimated_blocks, desc="Encoding blocks"):
            #encode blocks
            token_block = encode_block(block)
            #add tokens to a json file as you go
            #this way dont have to encode entire file for a result
            for tokens in token_block:
                json.dump(tokens, f)
                f.write("\n")

    