import sqlite3
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score


import Seq2Seq as s2s


pretrained_model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
model = BertModel.from_pretrained(pretrained_model_name)
vocab = tokenizer.vocab
output_dim = len(vocab)


def train(model, SRC, TRG, clip, device, nth_epoch, train=True):
    model.train() if train else model.eval()
    teacher_forcing_ratio = 0.6 if train else 0

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epoch_loss = 0
    epoch_bleu_score = 0
    batch_size = 128
    n_ite = len(TRG)/batch_size
    print(n_ite)

    for i in range(len(TRG))[::batch_size]:
        src_string = SRC[i:i+batch_size]
        trg_string = TRG[i:i+batch_size]

        tokenized_trg = tokenizer(trg_string, padding=True, return_tensors='pt')
        trg_ids = tokenized_trg['input_ids']
        trg = trg_ids.permute(1,0).to(device) # [trg len(ids), batch size]

        optimizer.zero_grad()
        output, output_index = model(src_string, trg, batch_size, teacher_forcing_ratio, nth_epoch)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        new_trg = trg[1:].reshape(-1)
        loss = criterion(output, new_trg)
        epoch_loss += loss.item()

        BLEU_SCORE = 0
        for i, line in enumerate(output_index):
            output_string = tokenizer.convert_ids_to_tokens(line)
            trg_string = tokenizer.convert_ids_to_tokens(trg_ids[i])
            print(f"output: {output_string}\ntrg:{trg_string}")
            score = bleu_score(output_string, trg_string)
            print(f"score: {score}")
            BLEU_SCORE += score

        epoch_bleu_score += BLEU_SCORE/batch_size

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


    return epoch_loss/n_ite


def draw(trainloss, validloss):
    writer = SummaryWriter()
    for i, (tra, val) in enumerate(zip(trainloss, validloss)):
        writer.add_scalar('train_loss', tra, i)
        writer.add_scalar('valid_loss', val, i)
    writer.close()


if '__main__' == __name__:
    conn = sqlite3.connect('aibo_backup.db')
    c = conn.cursor()
    # c.execute('select title from parsed_news where title is not Null')
    # title = c.fetchall()
    # trg = [t[0].strip().replace('\n', '') for t in title]
    # trg = []
    # for t in title:
    #     t = re.sub(r'《|【|】|》|！|？|「|」', '', t[0].strip())
    #     trg.append(t)
    c.execute('select article from parsed_news where article is not Null')
    articles = c.fetchall()
    articles = [a[0] for a in articles]
    src, trg = [],[]

    for article in articles:
        sentence = article.replace('，',' ').replace('。', ' ').split()
        print(f"len sentence:{len(sentence)}")
        src += sentence[:-1]
        trg += sentence[1:]

    print(f"len src:{len(src)}")
    print(f"len trg:{len(trg)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hid_dim = 256
    n_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    clip = 1

    encoder = s2s.Encoder(hid_dim, n_layers, enc_dropout)
    decoder = s2s.Decoder(hid_dim, n_layers, dec_dropout)
    model = s2s.Seq2Seq(encoder, decoder, device).to(device)

    n_epoch = 30
    best_valid_loss = float('inf')
    trainloss = []
    validloss = []

    for i in range(n_epoch):
        train_loss = train(model, src[:25600], trg[:25600], clip, device, i)
        valid_loss = train(model, src[25600:32000], trg[25600:32000], clip, device, i, train=False)

        trainloss.append(train_loss)
        validloss.append(valid_loss)

        print(f"epoch: {i}\ntrain loss: {train_loss}| valid loss: {valid_loss}")
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_valid_loss_model.pt')

    draw(trainloss, validloss)

    # model.load_state_dict(torch.load('./best_valid_loss_model.pt'))
    # test_loss = train(model, src[:600], trg[:600], clip, device, 1, train=False)
