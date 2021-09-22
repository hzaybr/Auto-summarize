import random
import numpy as np

import spacy
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

pretrained_model_name = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
model = BertModel.from_pretrained(pretrained_model_name)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(src)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(0)  # input = [1, batch size]
        print(f"emb inp:{self.embedding(inputs).shape}")
        embedded = self.dropout(self.embedding(inputs))
        output, (hidden, cell) = self.rnn(embedded,(hidden, cell))
        print(output.shape)
        prediction = self.fc_out(output.squeeze(0))
        print(prediction.shape)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        tokenized_str = tokenizer(src, padding=True, return_tensors='pt')
        src = model(**tokenized_str, output_hidden_states=True)
        src = src[0].permute(1,0,2)
        src = src.to(self.device)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)


        inputs = trg[0,:]
        output_index = np.zeros([batch_size, trg_len])
        output_index = output_index.astype('int64')

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            best_predictions = output.argmax(1)
            print(best_predictions)
            for i, prediction in enumerate(best_predictions):
                if prediction.item() == 3:
                    break
                else:
                    output_index[i][t-1] = prediction.item()

            inputs = trg[t] if teacher_force else best_predictions

        return outputs, output_index

