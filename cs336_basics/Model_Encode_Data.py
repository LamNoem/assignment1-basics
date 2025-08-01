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


def read_lines(file_path):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r', encoding="utf-8") as file, tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading file") as pbar:
        for line in file:
            pbar.update(len(line.encode('utf-8')))
            yield line

def encode_wrapper(args):
    tokenizer, line = args
    return tokenizer.encode(line)

from multiprocessing import Pool, cpu_count

def parallel_encode(tokenizer, file_path):
    lines = read_lines(file_path)
    args = ((tokenizer, line) for line in lines)

    with Pool() as pool:
        for tokens in tqdm(pool.imap(encode_wrapper, args, chunksize=100), desc="Tokenizing", unit="lines"):
            yield tokens

#--------------------------------------------------
global_tokenizer = None

def init_tokenizer():
    global global_tokenizer
    from BPETokenizer import BPETokenizer
    global_tokenizer = BPETokenizer.from_files(
        "data/owt_32k_merges_dict.json",
        "data/owt_32k_vocab.json",
        "data/owt_32k_merges.json"
    )

def count_lines(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return sum(1 for _ in f)


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


def encode_block(lines):
    from BPETokenizer import BPETokenizer
    tokenizer = BPETokenizer.from_files(
        "data/owt_32k_merges_dict.json",
        "data/owt_32k_vocab.json",
        "data/owt_32k_merges.json"
    )
    print("encoding block")
    return [tokenizer.encode(line) for line in lines]



def parallel_encode_block(file_path):
    total_lines = count_lines(file_path)
    estimated_blocks = total_lines // 1000 + 1
    blocks = read_blocks(file_path)

    with Pool(cpu_count(), initializer=init_tokenizer) as pool:
        for token_blocks in tqdm(pool.imap(encode_block, blocks, chunksize=10), total=estimated_blocks, desc="Tokenizing", unit="block"):
            for tokens in token_blocks:
                yield tokens



if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    #to train script
    tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")


    total_lines = count_lines("data/TinyStoriesV2-GPT4-train.txt")
    estimated_blocks = total_lines // 50 + 1

    with open("data/tiny_stories_ids_2.jsonl", "w") as f:
        for block in tqdm(read_blocks("data/TinyStoriesV2-GPT4-train.txt"), total=estimated_blocks, desc="Encoding blocks"):
            token_block = encode_block(block)
            for tokens in token_block:
                json.dump(tokens, f)
                f.write("\n")

    