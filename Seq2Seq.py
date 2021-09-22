import random
import numpy as np

import spacy
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

pretrained_model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
model = BertModel.from_pretrained(pretrained_model_name)
vocab = tokenizer.vocab
output_dim = len(vocab)
emb_dim = 768


class Encoder(nn.Module):
    def __init__(self, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()

    def forward(self, src_string, n, device):
        tokenized_src = tokenizer(src_string, padding=True, return_tensors='pt')

        # src = model(**tokenized_src, output_hidden_states=True)
        # src = src[0].permute(1,0,2).to(device)

        # if n < 5:
        #     src = model(**tokenized_src, output_hidden_states=True)
        #     src = src[0].permute(1,0,2).to(device)
        #     print(f"src shape: {src.shape}")
        # else:
        #     with torch.no_grad():
        #         src = model(**tokenized_src, output_hidden_states=True)
        #         src = src[0].permute(1,0,2).to(device)

        with torch.no_grad():
            src = model(**tokenized_src, output_hidden_states=True)
            src = src[0].permute(1,0,2).to(device)
            print(f"src shape: {src.shape}")

        src = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(src)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, hidden, cell, n, device):
        inputs = inputs.unsqueeze(0).cpu()

        # inputs = model(inputs)[0].to(device)

        # if n < 5:
        #     inputs = model(inputs)[0].to(device)
        # else:
        #     with torch.no_grad():
        #         inputs = model(inputs)[0].to(device)

        with torch.no_grad():
            inputs = model(inputs)[0].to(device)
            print(f"decoder inputs: {inputs.shape}")

        embedded = self.dropout(inputs)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_string, trg, batch_size, teacher_forcing_ratio, nth_epoch):

        hidden, cell = self.encoder(src_string, nth_epoch, self.device)

        trg_len = trg.shape[0]
        inputs = trg[0,:]
        print(f"trg len: {trg_len}")

        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)
        output_index = np.zeros([batch_size, trg_len]).astype('int64')

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell, nth_epoch, self.device)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            best_predictions = output.argmax(1)
            for i, prediction in enumerate(best_predictions):
                output_index[i][t-1] = prediction.item()

            inputs = trg[t] if teacher_force else best_predictions
            # if teacher_force:
            #     print(f"teach: {trg[t]}")
            # else:
            #     print(f"best predictions:{best_predictions}")
        return outputs, output_index

