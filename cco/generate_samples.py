import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from models import GRU, RNN 

train_path = os.path.join("C:\\Users\\Saber\\Desktop\\School\\IFT6135\\project2-acgs\\cco\\data\\ptb.train.txt")

torch.manual_seed(1111)

def make_sentence(sequence):
    sentence = ''
    for i in range(len(sequence)):
        sentence = sentence +  sequence[i] + ' '
    return sentence

def save_prediction(sequence, seq_length, file_name):
    with open(file_name, 'w') as text_file:
        for i in range(len(sequence)):
            sentence = ""
            for j in range(seq_length):
                sentence = sentence + sequence[i][j] + " "
            print(sentence + "\n", file = text_file)

def predict(id_2_word, seq_length=35, batch_size=20, load_GRU=True):
    if load_GRU:
        model = GRU(300, 1500, 35, 20, 10000, 2, 0.35)#(emb_size=350, hidden_size=1500, seq_len=35 batch_size=20, vocab_size=10000, num_layers=2, dp_keep_prob=0.35)
        model.load_state_dict(torch.load("model\\best_GRU.pt"))
        filename = "predictions\\GRU_"+ str(seq_length) + ".txt"
    else:
        model = RNN(200, 1500, 35, 20, 10000, 2, 0.35)
        model.load_state_dict(torch.load("model\\best_RNN.pt"))
        filename = "predictions\\RNN_"+ str(seq_length) + ".txt"
    hidden = model.init_hidden()

    random_input = torch.randint(10000, (batch_size,))
    samples = model.generate(random_input, hidden, seq_length, batch_size)

    sequence = [[" " for j in range(seq_length)] for i in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(seq_length):
            sequence[i][j] = id_2_word[samples[j, i].item()]
    save_prediction(sequence, seq_length, filename)
    
    return sequence
