import argparse
from TextRNN import TextRNN
import torch
import torch.nn.functional as func
import numpy as np
import json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dict', default='char_to_index.json', help='read char_dict from file')
    parser.add_argument('--load-model', default='model_1.pth', help='read model from file')
    parser.add_argument('--prediction-len',  type=int, default=250, help='length of a predicted text')
    opt = parser.parse_args()
    return opt


def generate(source_dict, load_model, prediction_len):
    print("Введите начало текста:")
    start_text = input()
    source = source_dict
    my_model = load_model
    char_to_idx = json.load(open(source))
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
    model.load_state_dict(torch.load(my_model))
    model.to(device)
    model.eval()

    temp = 0.3
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text
    _, hidden = model(train, hidden)
    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = func.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char
    print(predicted_text)


def main(opt):
    generate(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
