# TextGeneration-LSTM

#### Text generation based on the LSTM neural network 

<a href="https://colab.research.google.com/drive/1kZySrd2s4dKcAYu_Ozauu4Di39NisOy1?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Usage examples
Run code in the following cell:
```
!git clone https://github.com/blindsphynx/TextGeneration-LSTM.git
%cd TextGeneration-LSTM
```

Load your training dataset to the `TextGeneration-LSTM` folder if you want to train a model on your own data;

Then add the name of your training file after `--source` flag;

You can also choose sequence length `--seq-length`,  batch size `--batch` and number of training epochs `--epochs`.

To train a model, run the following cell:
```
!python train.py --source corpus.txt --seq-length 256 --batch 16 --epochs 10000 --save-dict-to char_to_index.json --save-model-to model_1.pth
```

For text generation you can choose a char dictionary, a pretrained model and a length of the generated sequence using 
`--source-dict`, `--load-model` and `--prediction-len` flags respectively.

To generate a text, run:
```
!python generate.py --source-dict char_to_index.json --load-model model_1.pth --prediction-len 350
```