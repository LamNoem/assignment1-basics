# import regex as re
# Automatically initializes dictionary values (e.g., counts).
from collections import defaultdict
# For reading/writing JSON files.
import json 
# Encodes/decodes binary data 
import base64
#Progress bar
from tqdm import tqdm
#File system utilities.
import os
#Extended regular expressions
import regex as re



class BPETokenizer:
    def __init__(self, merges_dict = {}, merges_list = [], vocab = {}, special_tokens=[]):
        # Dictionary to store merge operations: (token_id1, token_id2) -> new_token_id
        #Stores merge operations: which token pairs were merged and what new token ID they became.
        self.merges: dict[tuple[int, int], int] = merges_dict
        
        #Maps token IDs to their byte representations.
        self.vocab : dict[int, bytes] = vocab

        #Tracks the current size of the vocabulary (starts at 255 for byte values).
        #the original 0-255 byte representations of alphanumeric characters
        self.vocab_count :int = 255

        #Stores the sequence of merge operations in order.
        self.merges_list: list[tuple[bytes,bytes]] = merges_list

        #List of special tokens to be added to the vocabulary.
        self.special_tokens: list[str] = special_tokens

        #pretokenizing separately to work easier with chunks of text input and managing vocabulary size
        #Stores frequency counts of pre-tokenized strings.
        self.pretokens: dict[str, int] = defaultdict(int)

    
    #might cause error since where this function is used indices is entered as a tuple
    #Replaces all occurrences of a token pair in a sequence with a new token ID.
    def merge(self, indices: list[int], pair: tuple[int, int], new_index: int) -> tuple[int]:  
        #Iterates through the sequence and merges matching pairs.
        new_indices = []  
        i = 0  
        while i < len(indices):
            #if enough space, if identified pair
            #append new value (to new list) and move onto the next possible pair
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        #returns new sequence of token_ids (int represent. of bytes) once certain ids r merged and given a new id
        return tuple(new_indices)

        

    def __str__(self):
        return f"BPETokenizer(vocab={self.vocab}, merges={self.merges})"
    
    def pretokenize(self, text):
        #pretokenize
        #will pregroup a chunk of text and iterate within each group for byte pairs
        
        #Regex pattern captures contractions, words, numbers, punctuation, and whitespace.
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        #iterable of matches from text
        matches = re.finditer(PAT, text)
        #extracts the (string) matches into a list
        tokens = [match.group() for match in matches]

        #Counts frequency of each pretoken.
        #stores as member dictionary
        #self.pretokens -> pretoken (string), count (int)
        for pretoken in tokens:
            #pretoken_count[pretoken] += 1
            self.pretokens[pretoken] += 1

        return tokens
    
    #Loads tokenizer state from JSON files.
    @classmethod
    def from_files(cls, merges_dict_filepath, vocab_filepath, merges_filepath, special_tokens:str = None):
        #loads vocab
        with open(vocab_filepath, 'r') as vf:
            vocab_data = json.load(vf)
            vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}
        #loads merge dictionary
        with open(merges_dict_filepath, "r") as f:
            merges_data = json.load(f)
            merges_dict = {eval(k): v for k, v in merges_data.items()}
            
        #loads merge list
        with open(merges_filepath, 'r') as mf:
            merges_list = json.load(mf)
            voc_count = -1
        
        #start voc_count at -1 since 0 is included
        #get vocabulary count
        for key, item in vocab.items():
            voc_count += 1
         
        #add special tokens at the end
        if special_tokens != None: 
            for i in special_tokens:
                voc_count += 1
                vocab[voc_count] = i.encode("utf-8")

        return cls( merges_dict, merges_list, vocab, special_tokens)

    #trains tokenizer to reach a target vocab size
    def train_bpe(self, voc_size: int):

        #pretokenization must already be completed before this function is called
       

        #already default vocab
        #Initializes vocabulary with all initial byte values.
        #python uses integers to represent bytes
        self.vocab : dict[int, bytes] = { x : bytes([x]) for x in range(256)}
       
        #adding special tokens
        #Adds special tokens.
        for i in self.special_tokens:
            self.vocab_count += 1
            self.vocab[self.vocab_count] = i.encode("utf-8")


###################################################################################
        #pretokenize method
        #where pretokens r also a member variable
        #big text file is pretokenized (or pregrouped into words and such) a chunk at a time instead of all at once
        #then bpetokenizer is trained, by looking for pairs within the accumulated pretokens or groups
        #therefore instead of interating through an entire text for byte pairs, 
        #recognize words and such in the text as pretokens, store in a dictionary (key: words, value: freq)
        #then iterate through the key values to identify and store frequency of byte pairs

        #Encodes pretokens into byte sequences (tuple of integers)
        #pretokens dict[str, int of count]
        #pretoken_count_enc dict[tuple of byte sequence (ints), int of count]
        pretoken_count_enc = {tuple(map(int, k.encode("utf-8"))): v for k, v in self.pretokens.items()}
        #to keep track of number of merges in terminal
        co = 0

        #each loop is one merge, and one new vocab item
        #loop until vocab limit reached
        while self.vocab_count < voc_size:
            print("start merge"+str(co))
            co +=1
            #searching within each pretoken for byte pairs to merge
            #counts will store frequency of each unique byte pair
            counts = defaultdict(int)
            #key: pretoken, value: freq
            for key, value in pretoken_count_enc.items():
                #index1, index2 : byte pairs within pretoken
                #zip(...) returns interable of consecutive byte pairs (staggers two iterables, the keys, and the keys but starting at the 2nd key)
                
                #add each byte pair as a dictionary element, store the frequency
                for index1, index2 in zip(key, key[1:]):
                    counts[(index1,index2)] += value
            #if within each pretoken, no more merging is possible, training can't continue
            #unless you start merging pretokens
            if not counts:
                print("No more pairs to merge.")
                print(pretoken_count_enc)
                break  # Exit the training loop early

            #merging process
            #gets most frequent pair
            pair = max(counts, key=counts.get)
            index1, index2 = pair
            #increase vocab count by 1
            self.vocab_count += 1
            #add pair do merge dict
            self.merges[pair] = self.vocab_count
            #add to merge list
            self.merges_list.append((index1,index2))
            #add to vocab dict[int, byte]
            #the latest vocab_count is the new token id of the merged/concatenated byte pair
            self.vocab[self.vocab_count] = self.vocab[index1] + self.vocab[index2]
            
            #apply merge to pretokens
            # New optimized merge update

            #new encoded pretoken count dict
            new_enc = defaultdict(int)
            #seq: ints representing pretoken, freq: frequency
            for seq, freq in pretoken_count_enc.items():
                #if pair to be merged not in this sequence add it to new dictionary with the key unchanged, 
                # keep the frequency
                #if pair is indentified add the seq to the new dictionary 
                # but with the pair replaced with the new int value in the sequence
                if pair not in zip(seq, seq[1:]):
                    new_enc[seq] += freq  # unchanged
                else:
                    new_enc[self.merge(seq, pair, self.vocab_count)] += freq
            pretoken_count_enc = new_enc


#each merge took 24s for owt train
#now takes 17s
        
       
        #save merge list
        with open("data/owt_32k_merges_new.json", "w") as f:
            json.dump(self.merges_list, f)
        #save merge dict
        #json does not allow tuples as keys
        with open("data/owt_32k_merges_dict_new.json", "w") as f:
            json.dump({str(k): v for k, v in self.merges.items()}, f)

        # Convert bytes to base64 strings to ensure safe serialization
        vocab_serializable = {
            k: base64.b64encode(v.encode("utf-8") if isinstance(v, str) else v).decode("ascii")
            for k, v in self.vocab.items()
        }
        #Serializes vocabulary safely using base64.
        with open("data/owt_32k_vocab_new.json", "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2)


        return self.merges_list, self.vocab
    
    def encode(self, text :str):

        #pretokenize

        import re  # instead of regex
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""


        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, text)
        tokens = [match.group() for match in matches]
        #encode pretokens to math merge list
        #list of encoded pretokens, list of tuples
        pretoken_en= [tuple(map(int, k.encode("utf-8"))) for k in tokens]

        #merge within each pretoken
        #print(merges_dict)
        
        #within each pretoken/tuple, merge the tokens according to merge dictionary
        #then print the pretokens/tuples and their merges in order, for encoded text.
        for k,v in self.merges.items(): 
        
            for t in pretoken_en: 
                for index1, index2 in zip(t, t[1:]):  
                    p = index1, index2
                    #if is a merged pair
                    if k == p:
                        #print(k)
                        #print(p)
                        #print("#")
                        new_index = v
                        #replace pair in pretoken with new value
                        pretoken_en = [self.merge(t, p, new_index) for t in pretoken_en]

        #list of all int representations according to vocab and merge dict in a list to be printed
        pretoken_en = [x for tup in pretoken_en for x in tup]

        return pretoken_en
    

    

    

    
    
    def decode(self, ids: list[int]):
        string = ""
        for i in ids:
            #int,bytes
            default = "U+FFFD".encode("utf-8")
            #look up token id in vocab and return the bytes
            id_bts = self.vocab.get(i, default)
            #decode bytes and append to string
            id_str = id_bts.decode("utf-8",errors='replace')
        
            string += id_str
        return string
    
    def encode_iterable(self, file_path, file_name = None):

        file_size = os.path.getsize(file_path)  # total file size in bytes

        with open(file_path, 'r', encoding="utf-8") as file, tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding file") as pbar:
            # Read each line in the file
            for line in file:
            # Print each line
                tokens = self.encode(line)
                # for tok in tokens:
                #     yield tok
                yield tokens

                pbar.update(len(line.encode('utf-8')))  # bytes read


                
            
    def get_compression_ratio(string: str, indices: list[int]):
        num_bts = len(bytes(string, encoding="utf-8"))
        num_tok = len(indices)
        return num_bts/num_tok
    




       


 
        
                    



        

        

    





