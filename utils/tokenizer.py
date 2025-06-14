import json
import pickle

class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in text.split()]

    def decode(self, indices):
        return ' '.join([self.idx_to_word[idx] for idx in indices if idx != self.vocab['<pad>']])