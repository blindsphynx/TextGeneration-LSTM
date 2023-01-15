import argparse
import torch
import torch.nn as nn
import numpy as np
from TextRNN import TextRNN
from collections import Counter
import json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='corpus.txt', help='read train dataset from file')
    parser.add_argument('--seq-length', type=int, default=256, help='sequence length')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--save-dict-to', default='char_to_index.json', help='save char dict to file')
    parser.add_argument('--save-model-to', default='model_1.pth', help='save model to file')
    opt = parser.parse_args()
    return opt


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])
    return sequence, char_to_idx, idx_to_char


def get_batch(sequence, seq_len, batch_size):
    trains = []
    targets = []
    for _ in range(batch_size):
        batch_start = np.random.randint(0, len(sequence) - seq_len)
        chunk = sequence[batch_start: batch_start + seq_len]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


def train(source, seq_length, batch, epochs, save_dict_to, save_model_to):
    TRAIN_TEXT_FILE_PATH = source
    seq_len = seq_length
    batch_size = batch

    with open(TRAIN_TEXT_FILE_PATH, encoding='utf-8') as text_file:
        text_sample = text_file.readlines()
    text_sample = ' '.join(text_sample)

    sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)
    json.dump(char_to_idx, open(save_dict_to, 'w', encoding='utf-8'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )

    n_epochs = epochs
    loss_avg = []

    for epoch in range(n_epochs):
        model.train()
        train, target = get_batch(sequence, seq_len, batch_size)
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        hidden = model.init_hidden(batch_size)

        output, hidden = model(train, hidden)
        loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_avg.append(loss.item())
        if len(loss_avg) >= 50:
            mean_loss = np.mean(loss_avg)
            print(f'Loss: {mean_loss}')
            scheduler.step(mean_loss)
            loss_avg = []
            model.eval()
            print(f'Epoch: {epoch}')

    # Print model's state_dict
    print()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(model.state_dict(), save_model_to)

    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])
