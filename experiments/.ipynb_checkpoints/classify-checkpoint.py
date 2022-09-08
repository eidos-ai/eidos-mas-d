import former
from former import util
from former.util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import Counter

from functools import partial

# from torchtext import data, datasets, vocab
# from torchtext import data, datasets, vocab
from torchtext.datasets import IMDB
from torchtext import data
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

import numpy as np
import random

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
NUM_CLS = 2


def batch_sampler(batch_size, tokenizer, dataset_list):
    indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(dataset_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths 
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

def collate_batch(label_transform, text_transform, batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    padded = pad_sequence(text_list, padding_value=3.0)
    padded = padded.transpose_(0,1)
    return torch.tensor(label_list), padded 

def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    device = "cuda"
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load the IMDB data
    if arg.final:
        train, test = IMDB(split=('train', 'test'))
        
    else:
        tdata, _ = IMDB(split=('train', 'test'))
        tdata = list(tdata)
        random.shuffle(tdata)
        train, test = tdata[:int(len(tdata)*0.8)], tdata[int(len(tdata)*0.8):]
    
    train_list = list(train)
    test_list = list(test)
    
    
    counter = Counter()
    for (label, line) in train:
        counter.update(tokenizer(line))
    train_vocab = vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    train_vocab.set_default_index(train_vocab['<unk>'])
    
    label_transform = lambda x: 1 if x == 'pos' else 0
    text_transform = lambda x: [train_vocab['<BOS>']] + [train_vocab[token] for token in tokenizer(x)] + [train_vocab['<EOS>']]

    
    test_dataloader = DataLoader(list(test),
                                  collate_fn=partial(collate_batch, label_transform, text_transform),  
                                  batch_sampler=batch_sampler(arg.batch_size, tokenizer,test_list))
    
    print(f'- nr. of training examples {len(train_list)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_list)}')

    if arg.max_length < 0:
        mx = max([len(input[1]) for input in train])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    for e in range(arg.num_epochs):
        # Dataloaders have to be created inside epoch loop so that generator starts back again from zero
        train_dataloader = DataLoader(list(train),
                                      collate_fn=partial(collate_batch, label_transform, text_transform),  
                                      batch_sampler=batch_sampler(arg.batch_size, tokenizer,train_list))
        test_dataloader = DataLoader(list(test),
                                      collate_fn=partial(collate_batch, label_transform, text_transform),  
                                      batch_sampler=batch_sampler(arg.batch_size, tokenizer,test_list))
    
        print(f'\n epoch {e}')
        print("Train Step")
        model.train(True)
        pbar_train = tqdm.tqdm(total = len(train_list) / arg.batch_size)
        for i, (label, input) in enumerate(train_dataloader):
            opt.zero_grad()
            label = label.to(device)
            input = input.to(device)
        
            # print("label",type(label), label.shape) 
            # print("input", type(input), input.shape)
            
            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            seen += input.size(0)
            tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
            
            # pbar.update(arg.batch_size)
            # print("loss", loss)
            pbar_train.update(1)
        pbar_train.close()
        
        print("Evaluation Step")
        pbar_eval = tqdm.tqdm(total = len(test_list) / arg.batch_size)
        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0
            for i, (label, input) in enumerate(test_dataloader):
                label = label.to(device)
                input = input.to(device)
        
        
                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input).argmax(dim=1)
                
                tot += float(input.size(0))
                cor += float((label == out).sum().item())
                pbar_eval.update(1)
            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            tbw.add_scalar('classification/test-loss', float(loss.item()), e)
        pbar_eval.close()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
