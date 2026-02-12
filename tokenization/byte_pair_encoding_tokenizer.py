from tokenization.tokenizer import Tokenizer


class BytePairEncodingTokenizer(Tokenizer):
    def __init__(self, text, max_merges=5000):
        self.text = text
        self.tokens = list(text.encode("utf-8"))
        self.merges = {}
        self._vocab_size = 256  # Start with byte-level tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # For decoding
        # Automatically build the vocabulary
        self.build_tokens(max_epochs=max_merges)

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def build_tokens(self, max_epochs=5000):
        epoch = 0
        new_tokens = self.tokens
        merges = {}
        while epoch < max_epochs:
            stats = self.get_stats(new_tokens)
            top_pair = max(stats, key=stats.get)
            recurrence = stats[top_pair]
            if recurrence < 2:
                print(f"last recurrence is only: {recurrence} for pair {top_pair}. Quitting")
                break
            new_idx = 256 + epoch
            new_tokens = self.merge(new_tokens, top_pair, new_idx)
            merges[top_pair] = new_idx
            # Build vocab for decoding: new token = concatenation of pair
            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            epoch += 1

        print("number of epochs", epoch)
        print("number of new tokens", len(new_tokens))
        print("last ID reached: ", 256 + epoch)
        print(f"compression level: {len(self.tokens) / len(new_tokens):.2f}X")

        self.merges = merges
        self._vocab_size = 256 + epoch
        return new_tokens

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, text):
        """Encode text by converting to bytes and applying learned merges"""
        tokens = list(text.encode("utf-8"))
        # Apply merges in the order they were learned
        for pair, idx in self.merges.items():
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        """Decode token ids back to text"""
        # Convert each token id to its byte representation
        byte_array = b"".join(self.vocab[idx] for idx in ids)
        # Decode bytes to string
        return byte_array.decode("utf-8", errors="replace")
