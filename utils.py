from transformers import AutoTokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased"
    )
    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    sample = "จำเลยลักทรัพย์ผู้อื่นในเวลากลางคืน"
    tokens = tokenizer.encode(sample)
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens))