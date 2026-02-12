from tokenization.byte_pair_encoding_tokenizer import BytePairEncodingTokenizer

# Test with a small sample
test_text = "السلام عليكم ورحمة الله وبركاته"

print("Creating BPE tokenizer (building vocabulary with max 100 merges)...")
tokenizer = BytePairEncodingTokenizer(test_text, max_merges=100)

print(f"\nFinal vocab size: {tokenizer.vocab_size}")

# Test encode/decode
print("\n=== Testing encode/decode ===")
encoded = tokenizer.encode(test_text)
print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
print(f"Match: {decoded == test_text}")

# Test with new text
new_text = "بسم الله الرحمن الرحيم"
print(f"\n=== Testing with new text ===")
print(f"Original: {new_text}")
encoded = tokenizer.encode(new_text)
print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
print(f"Match: {decoded == new_text}")
