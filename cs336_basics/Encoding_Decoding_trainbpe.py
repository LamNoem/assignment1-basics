from BPETokenizer import BPETokenizer
import regex as re
from collections import defaultdict
import json
import base64


# #Tokenizing
# ############################################
#initialize tokenizer for training
tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
# ###################################################

# ###################################################
#training a tokenizer using a text

file_string = ""
# pretokenizing
with open('data/owt_train.txt', 'r', encoding='utf-8', errors='ignore') as file:
    print("Reading file in chunks...")
    for chunk in iter(lambda: file.read(1024 * 1024), ''):  # 1MB chunks
        
        tokenizer.pretokenize(str(chunk))
print("done reading")
# train
tokenizer.train_bpe( 32000)


##################################################
# encoding from a text

#initialize a tokenizer
tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")
# lst = tokenizer.encode(" ")
# print(lst)
# string = tokenizer.decode(lst)
# print(string)
# print("######")

# cot = 0

# for i in tokenizer.encode_iterable("data/story.txt"):
#          cot +=1
#          print(i)
#          print(tokenizer.decode([i]))
#          if cot > 40:
#              break
         

#########################################
#Encoding a text as a dataset
print(tuple(map(int, "the".encode("utf-8"))))
print(tokenizer.encode("the"))

# import json

# bad_lines = []

# with open("data/tiny_stories_ids.jsonl", "r") as f:
    
#     count = 0
#     for i, line in enumerate(f, start=1):

#         if count == 100:
#             break
        
#         try:
#             obj = json.loads(line)
#             print(obj)
#             print(tokenizer.decode(obj))
#             count = count + 1
#         except json.JSONDecodeError as e:
#             print(f"Error on line {i}: {e}")
#             bad_lines.append((i, line.strip()))
            
        