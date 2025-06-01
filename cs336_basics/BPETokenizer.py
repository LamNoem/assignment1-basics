import regex as re
from collections import defaultdict
import json
import base64



class BPETokenizer:
    def __init__(self, merges_dict = {}, merges_list = [], vocab = {}, special_tokens=[]):
        # Dictionary to store merge operations: (token_id1, token_id2) -> new_token_id
        self.merges: dict[tuple[int, int], int] = merges_dict
        # Dictionary to store vocabulary tokens: token_id -> bytes
        #consider special tokens
        #self.vocab : dict[int, bytes] = { x : bytes([x]) for x in range(256)}

        self.vocab : dict[int, bytes] = vocab

        self.vocab_count :int = 255

        self.merges_list: list[tuple[bytes,bytes]] = merges_list

        self.special_tokens: list[str] = special_tokens

        #pretokenizing separately to work easier with chunks of text input and managing vocabulary size
        self.pretokens: dict[str, int] = defaultdict(int)

    
    #might cause error since where this function is used indices is entered as a tuple
    def merge(self, indices: list[int], pair: tuple[int, int], new_index: int) -> tuple[int]:  # @inspect indices, @inspect pair, @inspect new_index
        """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
        new_indices = []  # @inspect new_indices
        i = 0  # @inspect i
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return tuple(new_indices)

        

    def __str__(self):
        return f"BPETokenizer(vocab={self.vocab}, merges={self.merges})"
    
    def pretokenize(self, text):
        #pretokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, text)
        tokens = [match.group() for match in matches]

        #pretoken_count = defaultdict(int)

        for pretoken in tokens:
            #pretoken_count[pretoken] += 1
            self.pretokens[pretoken] += 1

        #self.pretokens = pretoken_count.copy()
        #print(self.pretokens)
        return tokens
    
    @classmethod
    def from_files(cls, merges_dict_filepath, vocab_filepath, merges_filepath, special_tokens:str = None):
        with open(vocab_filepath, 'r') as vf:
            vocab_data = json.load(vf)
            vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

        with open(merges_dict_filepath, "r") as f:
            merges_data = json.load(f)
            merges_dict = {eval(k): v for k, v in merges_data.items()}
            

        with open(merges_filepath, 'r') as mf:
            merges_list = json.load(mf)
            voc_count = -1
        for key, item in vocab.items():
            voc_count += 1
         
        if special_tokens != None: 
            for i in special_tokens:
                voc_count += 1
                vocab[voc_count] = i.encode("utf-8")

        return cls( merges_dict, merges_list, vocab, special_tokens)

    #small scale first
    def train_bpe(self, voc_size: int):
        #read file for large scale

        #chunks

        #already default vocab
        self.vocab : dict[int, bytes] = { x : bytes([x]) for x in range(256)}
       
        #adding special tokens
        for i in self.special_tokens:
            self.vocab_count += 1
            self.vocab[self.vocab_count] = i.encode("utf-8")


###################################################################################
        #pretokenize method
        #where pretokens r also a member variable
        #big text file is pretokenized a chunk at a time instead of all at once
        pretoken_count_enc = {tuple(map(int, k.encode("utf-8"))): v for k, v in self.pretokens.items()}
        co = 0
        while self.vocab_count < voc_size:
            print("start merge"+str(co))
            co +=1
            counts = defaultdict(int)
            for key, value in pretoken_count_enc.items():
                
                
                for index1, index2 in zip(key, key[1:]):
                    counts[(index1,index2)] += value

            if not counts:
                print("No more pairs to merge.")
                print(pretoken_count_enc)
                break  # Exit the training loop early
            pair = max(counts, key=counts.get)
            index1, index2 = pair
            self.vocab_count += 1
            self.merges[pair] = self.vocab_count
            self.merges_list.append((index1,index2))
            self.vocab[self.vocab_count] = self.vocab[index1] + self.vocab[index2]
            
            #pretoken_count_enc = {self.merge(k,pair,self.vocab_count): v for k, v in pretoken_count_enc.items()}
            # New optimized merge update
            new_enc = defaultdict(int)
            for seq, freq in pretoken_count_enc.items():
                if pair not in zip(seq, seq[1:]):
                    new_enc[seq] += freq  # unchanged
                else:
                    new_enc[self.merge(seq, pair, self.vocab_count)] += freq
            pretoken_count_enc = new_enc


#each merge took 24s for owt train
#now takes 17s
        
       

        with open("data/owt_32k_merges.json", "w") as f:
            json.dump(self.merges_list, f)
        #json does not allow tuples as keys
        with open("data/owt_32k_merges_dict.json", "w") as f:
            json.dump({str(k): v for k, v in self.merges.items()}, f)

        # Convert bytes to base64 strings to ensure safe serialization
        vocab_serializable = {
            k: base64.b64encode(v.encode("utf-8") if isinstance(v, str) else v).decode("ascii")
            for k, v in self.vocab.items()
        }
        with open("data/owt_32k_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2)

        #to read it back
        #with open("vocab.json", "w", encoding="utf-8") as f:
            #json.dump(vocab_serializable, f, indent=2)

        return self.merges_list, self.vocab
    
    def encode(self, text :str):

        #pretokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, text)
        tokens = [match.group() for match in matches]
        #encode pretokens to math merge list
        pretoken_en= [tuple(map(int, k.encode("utf-8"))) for k in tokens]

        #merge within each pretoken
        #print(merges_dict)
        
        for k,v in self.merges.items(): 
        
            for t in pretoken_en: 
                for index1, index2 in zip(t, t[1:]):  
                    p = index1, index2
                    if k == p:
                        #print(k)
                        #print(p)
                        #print("#")
                        new_index = v
                        pretoken_en = [self.merge(t, p, new_index) for t in pretoken_en]
        pretoken_en = [x for tup in pretoken_en for x in tup]

        return pretoken_en
    
    def decode(self, ids: list[int]):
        string = ""
        for i in ids:
            #int,bytes
            default = "U+FFFD".encode("utf-8")
            id_bts = self.vocab.get(i, default)
            id_str = id_bts.decode("utf-8",errors='replace')
        
            string += id_str
        return string
    
    def encode_iterable(self, file_path, file_name = None):

        with open(file_path, 'r', encoding="utf-8") as file:
            # Read each line in the file
            for line in file:
            # Print each line
                tokens = self.encode(line)
                for tok in tokens:
                    yield tok

    def get_compression_ratio(string: str, indices: list[int]):
        num_bts = len(bytes(string, encoding="utf-8"))
        num_tok = len(indices)
        return num_bts/num_tok
    




       


 
        
                    



        

        

    





