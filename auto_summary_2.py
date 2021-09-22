import sqlite3
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score



model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-mnli')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')


def train(model, SRC, TRG, clip, device, train=True):
    model.train() if train else model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    epoch_loss = 0
    epoch_bleu_score = 0
    batch_size = 2
    n_ite = len(TRG)/batch_size
    print(f"n ite: {n_ite}")

    for i in range(len(TRG))[::batch_size]:
        src_string = SRC[i:i+batch_size]
        trg_string = TRG[i:i+batch_size]

        tokenized_src = tokenizer(src_string, max_length=1024, padding=True, return_tensors='pt')
        src_ids = tokenized_src['input_ids']

        output = model(**tokenized_src)[0]
        # print(f"src shape: {src_ids.shape}")
        # print(f"output shape: {output.shape}")

        # BLEU_SCORE = 0
        # for i, line in enumerate(output):
        #     src_len = len(src_string[i])
        #     print(f"src len: {src_len}")
        #     line = line.argmax(1)
        #     output_string = tokenizer.decode(line[1:])
        #     output_string_list = [o for o in output_string][:src_len]
        #     src_string_list = [s for s in src_string[i]]
        #     score = bleu_score(output_string_list, src_string_list)
        #     # print(f"output: {output_string}")
        #     # print(f"src: {src_string[i]}")
        #     # print(f"score: {score}")
        #     BLEU_SCORE += score
        # epoch_bleu_score += BLEU_SCORE/batch_size


        output_dim = output.shape[-1]
        re_output = output.reshape(-1, output_dim)
        re_src_ids = src_ids.reshape(-1)
        # print(f"output reshape: {re_output.shape}")
        # print(f"src reshape: {re_src_ids.shape}")

        optimizer.zero_grad()
        loss = criterion(re_output, re_src_ids)
        epoch_loss += loss.item()

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

    return epoch_loss/n_ite


def test(model, SRC, TRG, clip, device):
    model.eval()

    batch_size = 2
    n_ite = len(TRG)/batch_size
    print(f"n ite: {n_ite}")

    for i in range(len(TRG))[::batch_size]:
        src_string = SRC[i:i+batch_size]
        trg_string = TRG[i:i+batch_size]

        tokenized_trg = tokenizer(trg_string, max_length=1024, padding=True, return_tensors='pt')
        trg_ids = tokenized_trg['input_ids']
        trg = trg_ids.to(device)
        trg_len = len(trg_ids[-1].numpy())

        tokenized_src = tokenizer(src_string, max_length=1024, padding=True, return_tensors='pt')
        src_ids = tokenized_src['input_ids']

        output_ids = model.generate(src_ids, num_beams=4, max_length=trg_len, early_stopping=True)

        for i in range(batch_size):
            del_index = np.where(output_ids[i]<3)
            output = np.delete(output_ids[i], del_index)
            output_string = tokenizer.decode(output)
            print(f"output: {output_string}")
            print(f"trg: {trg_string[i]}")


def draw(trainloss, validloss):
    writer = SummaryWriter()
    for i, (tra, val) in enumerate(zip(trainloss, validloss)):
        writer.add_scalar('train_loss', tra, i)
        writer.add_scalar('valid_loss', val, i)
    writer.close()


if '__main__' == __name__:
    conn = sqlite3.connect('aibo_backup.db')
    c = conn.cursor()
    c.execute('select title from parsed_news where title is not Null')
    title = c.fetchall()
    trg = [t[0].strip().replace('\n', '') for t in title]
    # trg = []
    # for t in title:
    #     t = re.sub(r'《|【|】|》|！|？|「|」', '', t[0].strip())
    #     trg.append(t)
    c.execute('select article from parsed_news where article is not Null')
    articles = c.fetchall()
    src = [a[0].strip().replace('\n', '')[:350] for a in articles]


    print(f"len src:{len(src)}")
    print(f"len trg:{len(trg)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip = 1

    n_epoch = 5
    best_valid_loss = float('inf')
    trainloss = []
    validloss = []

    for i in range(n_epoch):
        train_loss = train(model, src[:500], trg[:500], clip, device)
        valid_loss = train(model, src[500:600], trg[500:600], clip, device, train=False)

        trainloss.append(train_loss)
        validloss.append(valid_loss)

        print(f"epoch: {i}\ntrain loss: {train_loss}| valid loss: {valid_loss}")
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_valid_loss_model.pt')

    draw(trainloss, validloss)

    model.load_state_dict(torch.load('./best_valid_loss_model.pt'))
    test(model, src[600:], trg[600:], clip, device)
