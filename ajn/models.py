import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from datetime import datetime
import sys
import matplotlib.pyplot as plt

from numpy import linalg as LA

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using GPU: " + str(0))
    device = torch.device("cuda")
    torch.cuda.set_device(0)
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return
    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Problem 1

class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.minibatch_grad_rnn = RNN_BPTT(vocab_size, hidden_size)
        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
        self.init_weights()
        self.dropout = nn.Dropout(1 - dp_keep_prob)
        self.tanh = nn.Tanh()

    def init_weights(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.Wx_linear = clones(nn.Linear(self.hidden_size, self.hidden_size,
                                          bias=True), self.num_layers - 1)
        self.Wx_linear.insert(0, nn.Linear(self.emb_size, self.hidden_size,
                                           bias=False))
        self.Wh_linear = clones(nn.Linear(self.hidden_size, self.hidden_size,
                                          bias=True), self.num_layers)
        self.Wy_linear = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.Wy_linear.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.Wy_linear.bias, val=0)
        nn.init.uniform_(self.Wx_linear[0].weight, a=-math.sqrt(1 / self.hidden_size),
                         b=math.sqrt(1 / self.hidden_size))

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        init_hidden_state = nn.Parameter(
            torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32))
        return init_hidden_state  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             dtype=torch.float32).to(device)
        word_embeddings = self.embedding(inputs)  # (seq_len, batch_size, emb_size)

        for time_idx in range(self.seq_len):
            x = self.dropout(word_embeddings[time_idx, :, :])

            for layer_idx in range(self.num_layers):
                cur_hidden = hidden[layer_idx, :, :].clone()
                x = self.tanh(self.Wx_linear[layer_idx](x)
                              + self.Wh_linear[layer_idx](cur_hidden))
                hidden[layer_idx, :, :] = x  # (batch_size, hidden_size)
                x = self.dropout(x)

            y_t = self.Wy_linear(x)
            logits[time_idx, :, :] = y_t

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        return samples


# Problem 2
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, input_bias=True):  # , batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_bias = input_bias
        # self.batch_size = batch_size

        self.lin_Wz = nn.Linear(input_size, hidden_size, input_bias)
        self.lin_Wr = nn.Linear(input_size, hidden_size, input_bias)
        self.lin_Wh = nn.Linear(input_size, hidden_size, input_bias)

        self.lin_Ur = nn.Linear(hidden_size, hidden_size)
        self.lin_Uz = nn.Linear(hidden_size, hidden_size)
        self.lin_Uh = nn.Linear(hidden_size, hidden_size)

        # Activation functions
        self.activation_z = nn.Sigmoid()
        self.activation_r = nn.Sigmoid()
        self.activation_h = nn.Tanh()

    def forward(self, input, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input, produced by the previous layer.
                            shape: (batch_size, input_size)
            - hidden: The initial hidden state for the GRU Cell.
                            shape: (batch_size, hidden_size)
        Returns:
            - The final hidden state of the GRU Cell.
                This will be used as the initial hidden state for all the
                mini-batches in an epoch, except for the first, where the return
                value of self.init_hidden will be used.
                See the repackage_hiddens function in ptb-lm.py for more details,
                if you are curious.
                        shape: (batch_size, hidden_size)
        """

        r = self.lin_Wr(input) + self.lin_Ur(hidden)
        r = self.activation_r(r)

        z = self.lin_Wz(input) + self.lin_Uz(hidden)
        z = self.activation_z(z)

        h = self.lin_Wh(input) + self.lin_Uh(r * hidden)
        h = self.activation_h(h)

        new_hidden = (1 - z) * hidden + z * h

        return new_hidden

    def init_weights_uniform(self):
        k = 1 / math.sqrt(self.hidden_size)

        nn.init.uniform_(self.lin_Wz.weight, -k, k)
        nn.init.uniform_(self.lin_Wr.weight, -k, k)
        nn.init.uniform_(self.lin_Wh.weight, -k, k)

        if self.input_bias:
            nn.init.uniform_(self.lin_Wz.bias, -k, k)
            nn.init.uniform_(self.lin_Wr.bias, -k, k)
            nn.init.uniform_(self.lin_Wh.bias, -k, k)

        nn.init.uniform_(self.lin_Uz.weight, -k, k)
        nn.init.uniform_(self.lin_Ur.weight, -k, k)
        nn.init.uniform_(self.lin_Uh.weight, -k, k)

        nn.init.uniform_(self.lin_Uz.bias, -k, k)
        nn.init.uniform_(self.lin_Ur.bias, -k, k)
        nn.init.uniform_(self.lin_Uh.bias, -k, k)

    # def init_hidden(self):
    #     if torch.cuda.is_available():
    #         hidden = nn.Parameter(torch.zeros(self.batch_size, self.hidden_size).cuda())
    #     else:
    #         hidden = nn.Parameter(torch.zeros(self.batch_size, self.hidden_size))
    #
    #     return hidden


class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.minibatch_grad_rnn = Gru_BPTT(vocab_size, hidden_size)
        # TODO ========================
        self.init_weights_uniform()
        self.dropout = nn.Dropout(1 - dp_keep_prob)


    def init_weights_uniform(self):
        # TODO ========================
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.gru_cells = clones(GRUCell(self.hidden_size, self.hidden_size, input_bias=True),
                                self.num_layers - 1)  # should other linear cells have a bias?
        self.gru_cells.insert(0, GRUCell(self.emb_size, self.hidden_size, input_bias=False))
        self.Wy_linear = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.Wy_linear.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.Wy_linear.bias, val=0)
        for i in range(self.num_layers):
            self.gru_cells[i].init_weights_uniform()

    def init_hidden(self):
        # TODO ========================
        init_hidden_state = nn.Parameter(
            torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32))
        return init_hidden_state  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             dtype=torch.float32).to(device)
        word_embeddings = self.embedding(inputs)  # (seq_len, batch_size, emb_size)

        for time_idx in range(self.seq_len):
            x = self.dropout(word_embeddings[time_idx, :, :])

            for layer_idx in range(self.num_layers):
                cur_hidden = hidden[layer_idx, :, :].clone()
                x = self.gru_cells[layer_idx](x, cur_hidden)
                hidden[layer_idx, :, :] = x  # (batch_size, hidden_size)
                x = self.dropout(x)

            y_t = self.Wy_linear(x)
            logits[time_idx, :, :] = y_t
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.
We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.
The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).
These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.
The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""


# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill

        # linear_list = clones(nn.Linear(n_units, n_units, bias=True), 4)
        self.query_linear = nn.Linear(n_units, n_units, bias=True)
        self.key_linear = nn.Linear(n_units, n_units, bias=True)
        self.value_linear = nn.Linear(n_units, n_units, bias=True)
        self.output_linear = nn.Linear(n_units, n_units, bias=True)

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        batch_size = query.size(0)

        # run first set of linear layers, then reshape query, key and values into multiple heads
        query = self.query_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # perform scaled-dot product attention
        A_presoftmax = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
            self.d_k)  # dim (batch_size, n_heads, seq_len, seq_len)
        if not mask is None:
            mask = mask.unsqueeze(1)  # add dimension of 1 at loc of n_heads to allow broadcasting
            A_presoftmax = A_presoftmax.masked_fill(mask == 0, -10 ** 9)

        A = F.softmax(A_presoftmax, dim=-1)  # apply softmax on n_heads
        if self.dropout:
            A = self.dropout(A)

        H = torch.matmul(A, value)  # dim (batch_size, n_heads, seq_len, d_k)

        # concatentate heads
        H_concat = H.transpose(1, 2).contiguous().view(batch_size, -1, self.n_units)

        # run output linear layer.
        attention_values = self.output_linear(H_concat)

        return attention_values  # size: (batch_size, seq_len, self.n_units)


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Problem 5.2
##############################################################################
#
# Average gradient of the loss per time-step
#
##############################################################################

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs

    def forward(self, x, tau = 1.0):
        e = np.exp( np.array(x) / tau )
        return e / np.sum( e )

class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x)

    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2


class RNNLayer:

    def __init__(self):
        self.mulGate = MultiplyGate()
        self.addGate = AddGate()
        self.activation = Tanh()

    def forward(self, x, prev_s, U, W, V):
        self.mulu = self.mulGate.forward(U, x)
        self.mulw = self.mulGate.forward(W, prev_s)
        self.add = self.addGate.forward(self.mulw, self.mulu)
        self.s = self.activation.forward(self.add)
        self.mulv = self.mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = self.mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = self.activation.backward(self.add, ds)
        dmulw, dmulu = self.addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = self.mulGate.backward(W, prev_s, dmulw)
        dU, dx = self.mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)


class RNN_BPTT:

    def __init__(self, word_dim, hidden_dim=100, bp_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bp_truncate = bp_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.num_examples= 0
        self.losses = []
        self.time_eval_loss=5
        self.learning_rate=0.005
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.grad_arr = []
        self.grad_arr_norm = []
        self.initialized = False

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.s
            layers.append(layer)
        return layers

    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def calculate_loss(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            dmulv = output.diff(layers[t].mulv, y[t])
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)
            for i in range(t - 1, max(-1, t - self.bp_truncate - 1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i - 1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU, dW, dV)

    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
        return [dU, dW, dV]

    def save_gradients(self, mini_batch_size):
        if self.initialized:
            self.tdLdU = LA.norm([x / mini_batch_size for x in self.tdLdU])
            self.tdLdV = LA.norm([x / mini_batch_size for x in self.tdLdV])
            self.tdLdW = LA.norm([x / mini_batch_size for x in self.tdLdW])
            self.grad_arr.append([self.tdLdU, self.tdLdW, self.tdLdV])
            print("Average gradient dL/dU,dL/dW,dL/dV: ", [self.tdLdU, self.tdLdW, self.tdLdV])

    def init_grad(self, sgd_vector):
        if self.initialized == False:
            self.tdLdU = np.zeros_like(sgd_vector[0])
            self.tdLdW = np.zeros_like(sgd_vector[1])
            self.tdLdV = np.zeros_like(sgd_vector[2])
            self.initialized = True
        self.tdLdU += sgd_vector[0]
        self.tdLdW += sgd_vector[1]
        self.tdLdV += sgd_vector[2]

    def train(self, rep_tensor, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, num_steps=20):
        num_examples_seen = 0
        losses = []
        initialized = False
        for epoch in range(nepoch):
            X = rep_tensor[:, epoch * num_steps:(epoch + 1) * num_steps]
            Y = rep_tensor[:, epoch * num_steps + 1:(epoch + 1) * num_steps + 1]
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                if (self.initialized):
                    tdLdU = [x / num_examples_seen for x in tdLdU]
                    tdLdV = [x / num_examples_seen for x in tdLdV]
                    tdLdW = [x / num_examples_seen for x in tdLdW]
                    self.grad_arr.append([tdLdU, tdLdW, tdLdV])
                    print("Average gradient dL/dU,dL/dW,dL/dV: ", [tdLdU, tdLdW, tdLdV])
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                dU, dW, dV = self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
                if initialized == False:
                    tdLdU = np.zeros_like(dU)
                    tdLdV = np.zeros_like(dV)
                    tdLdW = np.zeros_like(dW)
                    initialized = True
                tdLdU += dU
                tdLdV += dV
                tdLdW += dW
        return losses

class Gru_BPTT:

    def __init__(self, word_dim, hidden_dim = 30, bptt_truncate = 2 ):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        # Random asisgn weight
        self.U = np.random.uniform( -np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim) )
        self.W = np.random.uniform( -np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, hidden_dim) )
        self.V = np.random.uniform( -np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim) )
        self.b = np.zeros( (3, hidden_dim) )
        self.c = np.zeros( word_dim )

        self.num_examples= 0
        self.losses = []
        self.time_eval_loss=5
        self.learning_rate=0.005
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.grad_arr = []
        self.grad_arr_norm = []
        self.initialized = False
        pass

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        z = np.zeros((T + 1, self.hidden_dim))
        r = np.zeros((T + 1, self.hidden_dim))
        h = np.zeros((T + 1, self.hidden_dim))
        s = np.zeros((T + 1, self.hidden_dim))
        o= np.zeros((T, self.word_dim))

        for t in range(T):
            z[t]=self.sigmoid.forward(self.U[0,:,x[t]] + self.W[0].dot(s[t-1]) + self.b[2])
            r[t]=self.sigmoid.forward(self.U[1,:,x[t]] + self.W[1].dot(s[t-1]) + self.b[1])
            h[t]=np.tanh( self.U[2,:,x[t]] + self.W[2].dot(s[t-1]*r[t]) + self.b[0])
            s[t]=(1-z[t])*h[t]+z[t]*s[t-1]
            o[t]=self.softmax.forward( self.V.dot(h[t]) + self.c)
        return [z, r, h, s, o]

    def predict( self, x):
        z, r, h, s, o= self.forward_propagation( x )
        return np.argmax(o , axis = 1)

    def calculate_total_loss( self, x, y):
        L=0.0
        # For each sentences
        N = np.sum( ( len(y_i) for y_i in y) )
        for i in range( len(y) ):
            z, r, h, s, o = self.forward_propagation( x[i] )
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1* np.sum(np.log(correct_word_predictions))
        return L

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        z, r, h, s, o = self.forward_propagation(x)

        # Then we need to calculate the gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)
        dLdc = np.zeros(self.c.shape)

        delta_o = o
        delta_o[ np.arange(len(y)), y ] -= 1.0

        for t in np.arange(T)[::-1]:
            dLdV += np.outer( delta_o[t], s[t].T )
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            dLdc += delta_o[t]
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:

                dLdW[0] += np.outer(delta_t, s[bptt_step-1])
                dLdU[0,:,x[bptt_step]] += delta_t
                dLdb[0] += delta_t

                dLdr = self.W[0].T.dot(delta_t) * (s[bptt_step-1])
                dLdW[1] += np.outer( dLdr*r[bptt_step]*(1-r[bptt_step]), s[bptt_step-1] )
                dLdU[1,:,x[bptt_step]] += dLdr*r[bptt_step]*(1-r[bptt_step])
                dLdb[1] += dLdr * r[bptt_step] * (1-r[bptt_step])

                if bptt_step>=1:
                    dLdz = self.W[0].T.dot(delta_t) * r[bptt_step] * (s[bptt_step-2]-h[bptt_step])
                    dLdW[2] += np.outer( dLdz * z[bptt_step] * (1-z[bptt_step]), s[bptt_step-1] )
                    dLdU[2,:,x[bptt_step] ] += dLdz * z[bptt_step] * (1-z[bptt_step])
                    dLdb[2] += dLdz * z[bptt_step] * (1-z[bptt_step])

        return [ dLdU, dLdV, dLdW, dLdb, dLdc ]

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW, dLdb, dLdc = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        self.b -= learning_rate * dLdb
        self.c -= learning_rate * dLdc
        return dLdU, dLdV, dLdW, dLdb, dLdc

    def init_grad(self, sgd_vector):
        if self.initialized == False:
            self.tdLdU = np.zeros_like(sgd_vector[0])
            self.tdLdV = np.zeros_like(sgd_vector[1])
            self.tdLdW = np.zeros_like(sgd_vector[2])
            self.tdLdb = np.zeros_like(sgd_vector[3])
            self.tdLdc = np.zeros_like(sgd_vector[4])
            self.initialized = True
        self.tdLdU += sgd_vector[0]
        self.tdLdV += sgd_vector[1]
        self.tdLdW += sgd_vector[2]
        self.tdLdb += sgd_vector[3]
        self.tdLdc += sgd_vector[4]

    def save_gradients(self, mini_batch_size):
        if self.initialized:
            self.tdLdU = LA.norm([x / mini_batch_size for x in self.tdLdU])
            self.tdLdV = LA.norm([x / mini_batch_size for x in self.tdLdV])
            self.tdLdW = LA.norm([x / mini_batch_size for x in self.tdLdW])
            self.tdLdb = LA.norm([x / mini_batch_size for x in self.tdLdb])
            self.tdLdc = LA.norm([x / mini_batch_size for x in self.tdLdc])
            self.grad_arr.append([self.tdLdU, self.tdLdV, self.tdLdW, self.tdLdb, self.tdLdc])
            print("Average gradient dL/dU,dL/dV,dL/dW,dL/db,dL/dc: ", [self.tdLdU, self.tdLdV, self.tdLdW, self.tdLdb, self.tdLdc])

    def train(self,rep_tensor , learning_rate=0.003, nepoch=200, evaluate_loss_after=5, num_steps=20):
        losses = []
        initialized = False
        num_examples_seen = 0
        test = 0
        grad_step = 0
        for epoch in range(nepoch):
            X = rep_tensor[:, epoch * num_steps:(epoch + 1) * num_steps]
            Y = rep_tensor[:, epoch * num_steps + 1:(epoch + 1) * num_steps + 1]
            if epoch % evaluate_loss_after == 0:
                loss = self.calculate_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                if (initialized):
                    print("test", test)
                    test = 0
                    tdLdU = [x / num_examples_seen for x in tdLdU]
                    tdLdV = [x / num_examples_seen for x in tdLdV]
                    tdLdW = [x / num_examples_seen for x in tdLdW]
                    tdLdb = [x / num_examples_seen for x in tdLdb]
                    tdLdc = [x / num_examples_seen for x in tdLdc]
                    #grad_step = LA.norm([tdLdU, tdLdV, tdLdW, tdLdb, tdLdc])
                    #self.grad_arr_norm.append(grad_step)
                    self.grad_arr.append([tdLdU, tdLdV, tdLdW, tdLdb, tdLdc])
                    print("Average gradient normalized dL/dU, dL/dV, dL/dW, dL/db, dL/dc:",
                          [tdLdU, tdLdV, tdLdW, tdLdb, tdLdc])
                    #print("Gradient normalized:", grad_step)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                dLdU, dLdV, dLdW, dLdb, dLdc = self.sgd_step(X[i], Y[i], learning_rate)
                if initialized == False:
                    tdLdU = np.zeros_like(dLdU)
                    tdLdV = np.zeros_like(dLdV)
                    tdLdW = np.zeros_like(dLdW)
                    tdLdb = np.zeros_like(dLdb)
                    tdLdc = np.zeros_like(dLdc)
                    initialized = True
                tdLdU += dLdU
                tdLdV += dLdV
                tdLdW += dLdW
                tdLdb += dLdb
                tdLdc += dLdc
                num_examples_seen += 1
                test += 1
        return losses





