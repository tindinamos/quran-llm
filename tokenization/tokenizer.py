from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Base class for all tokenizers"""

    @abstractmethod
    def encode(self, text):
        """Encode text into a list of token ids"""
        pass

    @abstractmethod
    def decode(self, ids):
        """Decode a list of token ids back into text"""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """Return the vocabulary size"""
        pass
