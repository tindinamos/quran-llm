class ModelConfig:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
