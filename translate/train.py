# third-party import
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizer, BertModel
#from torch.utils.tensorboard import SummaryWriter


# local import
import s2s as s2s


def train(model, iterator, clip, SRC, TRG, train=True):
    model.train() if train else model.eval()
    teacher_forcing_ratio = 0.5 if train else 0

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    epoch_loss = 0
    epoch_bleu_score = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        batch_size = trg.shape[-1]

        src = src.cpu().detach().numpy()
        src = np.transpose(src)
        src_string = []
        for line in src:
            del_index = np.where(line<4)
            string = np.delete(line, del_index)
            string = ' '.join([SRC.vocab.itos[word] for word in string])
            src_string.append(string)

        optimizer.zero_grad()
        output, output_index = model(src_string, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        new_trg = trg[1:].view(-1)
        loss = criterion(output, new_trg)
        epoch_loss += loss.item()

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        trg_index = trg.cpu().detach().numpy()
        trg_index = np.transpose(trg_index)
        BLEU_SCORE = 0

        for i, line in enumerate(output_index):
            unk_index = np.where(line==0)
            line = np.delete(line, unk_index)
            output_string = [[TRG.vocab.itos[word] for word in line]]

            del_index = np.where(trg_index[i]<4)
            new_trg_idx = np.delete(trg_index[i], del_index)
            trg_string = [[[TRG.vocab.itos[index] for index in new_trg_idx]]]

            w = np.full(len(output_string), 1/len(output_string))
            score = bleu_score(output_string, trg_string, len(w), w)
            BLEU_SCORE += score

            # print(output_string)
            # print(trg_string)
            # print(score)

        epoch_bleu_score += BLEU_SCORE/batch_size


    return epoch_loss/len(iterator), epoch_bleu_score/len(iterator)

# def draw(trained, valid):
#     writer = SummaryWriter()
#     for i, (tra, val) in enumerate(zip(trained, valid)):
#         writer.add_scalar('train_loss', tra[0], i)
#         writer.add_scalar('valid_loss', val[0], i)
#         writer.add_scalar('train_bleu_score', tra[1], i)
#         writer.add_scalar('valid_bleu_score', val[1], i)
#     writer.close()

if '__main__' == __name__:
    spacy_ger = spacy.load('de')
    spacy_eng = spacy.load('en')

    def tokenizer_ger(text):
        return [tok.text for tok in spacy_ger.tokenizer(text)]


    def tokenizer_eng(text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]


    SRC = Field(tokenize = tokenizer_ger, lower=True,
                init_token='<sos>', eos_token='<eos>')
    TRG = Field(tokenize = tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

    print(f"Number of training examples : {len(train_data.examples)}")
    print(f"Number of validation examples : {len(valid_data.examples)}")
    print(f"Number of testing examples : {len(test_data.examples)}")

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    print(f"Source vocab dict : {len(SRC.vocab)}")
    print(f"Target vocab dict : {len(TRG.vocab)}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    encoder_emb_dim = 768
    decoder_emb_dim = 512
    hid_dim = 256
    n_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5

    encoder = s2s.Encoder(input_dim, encoder_emb_dim, hid_dim, n_layers, enc_dropout)
    decoder = s2s.Decoder(output_dim, decoder_emb_dim, hid_dim, n_layers, dec_dropout)
    model = s2s.Seq2Seq(encoder, decoder, device).to(device)

    n_epoch = 30
    clip = 1
    best_valid_loss = float('inf')
    best_bleu_score = 0
    trained = []
    valid = []

    for epoch in range(n_epoch):
        train_loss = train(model, train_iterator, clip, SRC, TRG)
        valid_loss = train(model, valid_iterator, clip, SRC, TRG, train=False)
        print(f"Epoch {epoch} :\nTrain (Loss, Score) was {train_loss} | Valid (Loss, Score) was {valid_loss}")
        trained.append(train_loss)
        valid.append(valid_loss)

        if valid_loss[0] < best_valid_loss:
            best_valid_loss = valid_loss[0]
            torch.save(model.state_dict(), 'best_valid_loss_model.pt')
        if valid_loss[1] > best_bleu_score:
            best_bleu_score = valid_loss[1]
            torch.save(model.state_dict(), 'best_bleu_score_model.pt')

    draw(trained, valid)
