from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Make sure punkt is downloaded
nltk.download("punkt")

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = len(self.word2idx)

        for sentence in sentence_list:
            if not isinstance(sentence, str):
                continue
            tokens = word_tokenize(sentence.lower())
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokens = word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def decode(self, indices):
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

