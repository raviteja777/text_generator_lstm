import os
import re
import torch


# file_path = "data/sample1.txt"
# inp_len = 4


class DataProcessor:


    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.inp_len = 0
        self.vocab = self.create_vocab()

    @staticmethod
    def create_vocab():
        chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        chars = chars + [' ', '.', ':', ';', '`', '"', "'"]
        chars = dict({(v, k + 1) for k, v in enumerate(chars)})
        chars['EOL'] = 0
        return chars

    # takes dirpath and creates embedding
    def create_dataset(self):
        embeddings = []
        for f in os.listdir(self.dir_path):
            print("process file: ", f)
            embeddings += self._embed_file_(os.path.join(self.dir_path, f))
        return self._transform_sequences_(embeddings)

    def _embed_file_(self, dat_file):
        embed = []
        dat = open(dat_file, 'r')
        # print('embedding file ',dat)
        for line in dat:
            line = line.strip().lower()
            # print(line)
            if len(line) >= 2:
                line = re.sub(r'\s+', ' ', line)
                em = []
                for ch in line:
                    if ch in self.vocab.keys():
                        em.append(self.vocab[ch])
                embed.append(em)
        return embed

    # pad 0s to make uniform length sequences
    # convert to tuple(feature tensor , label tensor)  and append to dataset
    def _transform_sequences_(self, embed):
        dataset = []
        max_len = max(map(len, embed))
        embed = [seq + (max_len - len(seq)) * [0] for seq in embed]
        for seq in embed:
            features = torch.tensor(seq[0:max_len - 1])
            label = torch.tensor(seq[1:])
            dataset.append((features, label))
        self.inp_len = max_len - 1
        print("===Dataset sample ====")
        print(*dataset[:5], sep="\n")
        return dataset

    def get_vocab(self):
        return self.vocab

    def get_inp_len(self):
        return self.inp_len
