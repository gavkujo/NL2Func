# train.py

import os
import json
import torch
import random
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

from models.transformer import MiniTransformer

# ─── Dataset ──────────────────────────────────────────────────────────────────

class NL2FuncDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # load lines (either JSONL or JSON list)
        raw = open(json_path, 'r', encoding='utf-8').read()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                self.examples = data
            else:
                raise ValueError
        except ValueError:
            # JSONL
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        entry = self.examples[idx]
        src = entry['input']
        tgt = entry['output']

        # tokenize (adds BOS/EOS)
        src_ids = [self.tokenizer.PieceToId('[BOS]')] + \
                  self.tokenizer.EncodeAsIds(src) + \
                  [self.tokenizer.PieceToId('[EOS]')]
        tgt_ids = [self.tokenizer.PieceToId('[BOS]')] + \
                  self.tokenizer.EncodeAsIds(tgt) + \
                  [self.tokenizer.PieceToId('[EOS]')]

        # truncate
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(list(src_batch), batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(list(tgt_batch), batch_first=True, padding_value=0)
    return src_batch, tgt_batch

# ─── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    # prepare paths
    os.makedirs(args.save_dir, exist_ok=True)
    sp_model = args.tokenizer + '.model'
    if not os.path.exists(sp_model):
        print("Training tokenizer…")
        os.system(
            f"python tokenizer/train_sentencepiece.py --data {args.data} "
            f"--prefix {args.tokenizer} --vocab_size {args.vocab_size}"
        )
    print("Loading tokenizer…")
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)

    # dataset
    full_dataset = NL2FuncDataset(args.data, sp, max_len=args.max_len)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # model
    model = MiniTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # optimizer + scheduler + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            # prepare decoder input and target
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            logits = model(src, decoder_input)
            logits = logits.reshape(-1, logits.size(-1))
            decoder_target = decoder_target.reshape(-1)

            loss = criterion(logits, decoder_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                logits = model(src, decoder_input)
                logits = logits.reshape(-1, logits.size(-1))
                decoder_target = decoder_target.reshape(-1)
                loss = criterion(logits, decoder_target)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NL2Func Transformer")
    parser.add_argument('--data', type=str, default='data/full_dataset.json')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/tokenizer')
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="Fraction of data to use for validation")
    args = parser.parse_args()
    train(args)
