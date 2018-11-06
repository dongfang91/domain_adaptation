import read_files as read
import numpy as np

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Post-Pad with zeroes.
    """
    x = []
    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(0)

    while len(x) < 356:
        x.append(4)
    return x

char,pos,unicate = read.load_hdf5("data/cvcolon_train_input",["char","pos","unic"])

char2int = read.readfrom_json("data/char2int")
int2char = {index:char for char, index in char2int.items() }
# print(char2int)
int2char = dict((c, i) for i, c in char2int.items())
sent = list()
sent_len = list()
for char_x_sent in char:  # 2637    8820     12760     ####2637     6183    3940     7140
    sent_single = [int2char[i] if i != 88 and i != 0 else ' ' for i in char_x_sent]
    sent.append(sent_single)


import torch
forward_flairTorch = torch.load("data/lm-news-english-forward-v0.2rc.pt")
dictionary = {k.decode('utf8'): v for k, v in forward_flairTorch['dictionary'].item2idx.items()}

char_new_input = [get_idx_from_sent(sent_txt,dictionary) for sent_txt in sent ]
print(char_new_input)
read.save_hdf5("data/cvcolon_train_input_flair",['char'],[np.asarray(char_new_input)],['int16'])








