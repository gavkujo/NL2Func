# tokenizer/train_sentencepiece.py

import json
import os
import sentencepiece as spm
from tempfile import NamedTemporaryFile

def load_texts(json_path):
    """
    Load all input and output texts from the dataset.
    Returns a list of strings.
    """
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        try:
            data = json.loads(raw)
            # If it's a list, treat as JSON list
            if isinstance(data, list):
                for e in data:
                    texts.append(e['input'])
                    texts.append(e['output'])
                return texts
        except json.JSONDecodeError:
            pass
    # If not a JSON list, treat as JSONL
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            texts.append(entry['input'])
            texts.append(entry['output'])
    return texts

def train_sentencepiece(dataset_path: str,
                        model_prefix: str = 'tokenizer/tokenizer',
                        vocab_size: int = 8000,
                        character_coverage: float = 1.0,
                        model_type: str = 'unigram'):
    """
    Train a SentencePiece model on the combined input/output texts.
    Saves tokenizer.model and tokenizer.vocab under the prefix.
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    texts = load_texts(dataset_path)

    # Write all texts to a temp file, one line per example
    with NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmp:
        for txt in texts:
            # Replace newlines to keep one-sentence-per-line
            line = txt.replace('\n', ' ')
            tmp.write(line + '\n')
        tmp_path = tmp.name

    # Train SentencePiece
    spm.SentencePieceTrainer.Train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    # Cleanup temp file
    os.remove(tmp_path)
    print(f"Trained SentencePiece model: {model_prefix}.model ({vocab_size} pieces)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument('--data', type=str, default='data/full_dataset.json',
                        help="Path to JSON dataset")
    parser.add_argument('--prefix', type=str, default='tokenizer/tokenizer',
                        help="Prefix for output model files")
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help="Vocabulary size")
    parser.add_argument('--model_type', type=str, default='unigram',
                        choices=['unigram', 'bpe', 'word', 'char'],
                        help="Type of SentencePiece model")
    args = parser.parse_args()

    train_sentencepiece(
        dataset_path=args.data,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )
