class OneLetterTokenizer:
    def __init__(self, text):
        chars = sorted(set(expand_tokens(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = dict(enumerate(chars))
        self.vocab_size = len(chars)

    def encode(self, text):
        i = 0
        total = []
        while i < len(text):
            token, i = next_token(i, text)
            total.append(self.char_to_idx[token])
        return total

    def decode(self, i):
        return "".join([self.idx_to_char[ii] for ii in i])


std_vocab = "0123456789|ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي ۩"
reveries_vocab = "ًٌٍَُِّْ"
numerics = "0123456789|"


def next_token(index, text):
    if text[index] in numerics:
        token = [
            text[index],
        ]
        j = index + 1
        while j < len(text) and text[index] in numerics:
            token.append(text[j])
            j += 1
            if text[j - 1] == "|":
                break
        return "".join(token), j
    elif text[index] in std_vocab:
        token = [
            text[index],
        ]
        j = index + 1
        while j < len(text) and text[j] in reveries_vocab:
            token.append(text[j])
            j += 1
        return "".join(token), j
    else:
        return text[index], index + 1


def expand_tokens(test_text):
    total = []
    i = 0
    while i < len(test_text):
        token, i = next_token(i, test_text)
        total.append(token)
    return total


def main():
    with open("data/quran-simple.txt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = OneLetterTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    test_text = "إِيَادْ إِبْرَاهِيمْ"
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
