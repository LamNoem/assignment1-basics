from BPETokenizer import BPETokenizer
import regex as re
from collections import defaultdict
import json
import base64


# #Tokenizing
# ############################################
tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
# ###################################################

# # with open('data/owt_train.txt', 'r') as file:
# #     print("reading file")
# #     file_string = file.read()
# # raised memory error so handle in chuncks instead


# ###################################################

# file_string = ""
# with open('data/owt_train.txt', 'r', encoding='utf-8', errors='ignore') as file:
#     print("Reading file in chunks...")
#     for chunk in iter(lambda: file.read(1024 * 1024), ''):  # 1MB chunks
        
#         tokenizer.pretokenize(str(chunk))
# print("done reading")
# tokenizer.train_bpe( 32000)



# ###############################################
# # print("self.merge")
# # print(tokenizer.merges)
# # print("self.vocab")
# # print(tokenizer.vocab)

#tokenizer = BPETokenizer.from_files("data/merges_dict.json", "data/vocab.json", "data/merges.json")

# print(tokenizer.vocab)
# print(tokenizer.vocab[256])

#ENCODING Decoding
##################################################
print("aayo")
tokenizer = BPETokenizer.from_files("data/owt_32k_merges_dict.json", "data/owt_32k_vocab.json", "data/owt_32k_merges.json")
lst = tokenizer.encode(" ")
print(lst)
string = tokenizer.decode(lst)
print(string)
print("######")

cot = 0

for i in tokenizer.encode_iterable("data/story.txt"):
         cot +=1
         print(i)
         print(tokenizer.decode([i]))
         if cot > 40:
             break