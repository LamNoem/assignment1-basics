import regex as re
import json
import base64
import torch
from collections import defaultdict

class BPETokenizer_Pytorch:
    def __init__(self, merges_dict={}, merges_list=[], vocab={}, special_tokens=[]):
        self.merges: dict[tuple[int, int], int] = merges_dict
        self.vocab: dict[int, torch.Tensor] = {x: torch.tensor([x], dtype=torch.uint8) for x in range(256)}
        self.vocab.update({k: torch.tensor(list(v.encode("utf-8")), dtype=torch.uint8) for k, v in vocab.items()})
        self.vocab_count: int = 255
        self.merges_list: list[tuple[int, int]] = merges_list
        self.special_tokens: list[str] = special_tokens
        self.pretokens: dict[str, int] = defaultdict(int)

        for token in self.special_tokens:
            self.vocab_count += 1
            self.vocab[self.vocab_count] = torch.tensor(list(token.encode("utf-8")), dtype=torch.uint8)

    def pretokenize(self, text):
        """Pre-tokenizes text using regex."""
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = [match.group() for match in re.finditer(PAT, text)]
        for pretoken in tokens:
            self.pretokens[pretoken] += 1
        return tokens

    def merge(self, seq: tuple[int], pair: tuple[int, int], new_index: int) -> tuple[int]:
        """Efficiently merges token pairs using PyTorch."""
        seq_tensor = torch.tensor(seq, dtype=torch.int32)
        merged_tensor = seq_tensor.clone()
        mask = (seq_tensor[:-1] == pair[0]) & (seq_tensor[1:] == pair[1])
        merged_tensor[:-1][mask] = new_index
        return tuple(merged_tensor.tolist())


    def train_bpe(self, voc_size: int):
        """Trains BPE using PyTorch-based optimizations."""
        pretoken_count_enc = {tuple(map(int, k.encode("utf-8"))): v for k, v in self.pretokens.items()}
        co = 0
        while self.vocab_count < voc_size:
            print("merge"+str(co))
            co+=1
            counts = defaultdict(int)
            for key, value in pretoken_count_enc.items():
                key_tensor = torch.tensor(key, dtype=torch.int32)
                pairs = torch.stack((key_tensor[:-1], key_tensor[1:]), dim=1).tolist()
                for pair in pairs:
                    counts[tuple(pair)] += value

            if not counts:
                print("No more pairs to merge.")
                break

            pair = max(counts, key=counts.get)
            index1, index2 = pair
            self.vocab_count += 1
            self.merges[pair] = self.vocab_count
            self.merges_list.append(pair)
            self.vocab[self.vocab_count] = torch.cat([self.vocab[index1].clone(), self.vocab[index2].clone()])


            # Updating pretoken counts efficiently
            new_enc = defaultdict(int)
            for seq, freq in pretoken_count_enc.items():
                if pair not in zip(seq, seq[1:]):
                    new_enc[seq] += freq
                else:
                    new_enc[self.merge(seq, pair, self.vocab_count)] += freq
            pretoken_count_enc = new_enc

        # Save merges list (Token Merges)
        with open("data/owt_32k_merges.json", "w") as f:
            json.dump(self.merges_list, f, indent=2)

        # Save merges dictionary (Token Pair Mappings)
        with open("data/owt_32k_merges_dict.json", "w") as f:
            json.dump({str(k): v for k, v in self.merges.items()}, f, indent=2)

        # Convert bytes in vocabulary to Base64 for safe storage
        vocab_serializable = {k: base64.b64encode(v.numpy().tobytes()).decode("ascii")  # Ensures proper handling of tensors
                                    for k, v in self.vocab.items()}

        # Save vocabulary
        with open("data/owt_32k_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2)


        return self.merges_list, self.vocab

    def encode(self, text: str):
        """Encodes input text using trained BPE merges."""
        tokens = self.pretokenize(text)

        # Convert tokens into numerical sequences
        pretoken_en = [tuple(map(int, k.encode("utf-8"))) for k in tokens]

        # Apply merges correctly without undefined `t`
        for k, v in self.merges.items():
            for seq in pretoken_en:
                pretoken_en = [self.merge(seq, k, v) if k in zip(seq, seq[1:]) else seq for seq in pretoken_en]

        # Flatten nested tuples into a list
        return [x for seq in pretoken_en for x in seq]

    def decode(self, ids: list[int]):
        """Decodes token IDs back into text."""
        decoded_string = []
        for i in ids:  # Iterate over `ids`
            tensor_bytes = self.vocab.get(i, torch.tensor(list("�".encode("utf-8")), dtype=torch.uint8))  # Use replacement character
            decoded_string.append(tensor_bytes.numpy().tobytes().decode("utf-8", errors="replace"))

        return "".join(decoded_string)

