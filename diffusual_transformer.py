import argparse
import os
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_bpe_tokenizer(words_file: str, vocab_size: int, save_path: str):
    """Train a BPE tokenizer on the provided words file and save it to save_path."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[MASK]"])
    tokenizer.train([words_file], trainer)
    tokenizer.save(save_path)
    print(f"Saved trained BPE tokenizer to {save_path}")
    return save_path

def load_tokenizer(path: str) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    return tok

def load_words_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf8') as f:
        words = [w.strip() for w in f if w.strip()]
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq

def encode_words_to_ids(words: List[str], tokenizer: Tokenizer, max_len: int, pad_id: int):
    """Encode each word into a fixed-length sequence of token ids (pad/truncate)."""
    encodings = []
    for w in words:
        enc = tokenizer.encode(w)
        ids = enc.ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [pad_id] * (max_len - len(ids))
        encodings.append(ids)
    return encodings

class BPEWordDataset(Dataset):
    def __init__(self, encoded_ids: List[List[int]]):
        self.encoded = encoded_ids

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long)

def make_linear_schedule(T: int, start: float = 0.0, end: float = 0.9):
    """Return alpha_t array of length T+1 (probability of keeping token at step t).
    alpha_0 = 1.0, alpha_T approx 1-end (so replacement prob increases to end).
    """
    alphas = [1.0]
    for t in range(1, T + 1):
        a = 1.0 - (t / T) * end
        alphas.append(max(0.0, a))
    return torch.tensor(alphas)

class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.emb = nn.Embedding(T + 1, dim)

    def forward(self, t: torch.LongTensor):
        return self.emb(t)

class SimpleDiffusionTransformer(nn.Module):
    """Diffusion Transformerm module
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, seq_len=8, T=200):
        super().__init__()

        # Model embeddings sizes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # nn.Embeddings convert discrete value to vector 
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_emb = TimeEmbedding(T, d_model)

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # взято из ориг. кода, тк с такой начальной развесовкой модель стабильнее сходиться
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.LongTensor, t: torch.LongTensor):
        batch, seq_len = x.shape
        assert seq_len == self.seq_len
        tok = self.token_emb(x) 

        # Position embedding for attention
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        
        pos = self.pos_emb(pos_ids)
        time = self.time_emb(t)
        time = time.unsqueeze(1).expand(-1, seq_len, -1)

        # Input transformer layer
        h = tok + pos + time
        h = self.transformer(h)
        logits = self.out_proj(h)
        return logits

def sample_from_model(model: nn.Module, tokenizer: Tokenizer, seq_len: int, T: int, alphas: torch.Tensor, device: torch.device, sample_steps=None):
    model.eval()
    if sample_steps is None:
        sample_steps = list(range(T, 0, -1))
    batch = 1
    vocab_size = len(tokenizer.get_vocab())
    xt = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)
    with torch.no_grad():
        for t in sample_steps:
            t_idx = torch.tensor([t], device=device, dtype=torch.long)
            logits = model(xt, t_idx)  # (1, seq_len, vocab)
            probs = F.softmax(logits, dim=-1)
            # Getting prediction on every step using random sampling due to logits
            xt = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch, seq_len)
    # decode first sequence
    ids = xt[0].tolist()
    # remove padding tokens
    if tokenizer.token_to_id('[PAD]') is not None:
        pad_id = tokenizer.token_to_id('[PAD]')
        ids = [i for i in ids if i != pad_id]
    text = tokenizer.decode(ids)
    return text, ids


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer setup
    if args.train_tokenizer:
        assert args.tokenizer_path is not None, "--tokenizer_path required when training tokenizer"
        train_bpe_tokenizer(args.words_file, args.vocab_size, args.tokenizer_path)

    assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), "Tokenizer file must exist (use --train_tokenizer to create)."
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    pad_id = tokenizer.token_to_id('[PAD]') if tokenizer.token_to_id('[PAD]') is not None else 0

    words = load_words_file(args.words_file)
    encoded = encode_words_to_ids(words, tokenizer, args.seq_len, pad_id)
    ds = BPEWordDataset(encoded)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SimpleDiffusionTransformer(vocab_size=vocab_size,
                                       d_model=args.d_model,
                                       n_heads=args.n_heads,
                                       n_layers=args.n_layers,
                                       seq_len=args.seq_len,
                                       T=args.T).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # warm start
    global_step = 0
    if args.resume_path is not None and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optim_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optim_state_dict'])
        if 'step' in ckpt:
            global_step = ckpt['step']
        print(f"Resumed from {args.resume_path} at step {global_step}")

    alphas = make_linear_schedule(args.T, start=0.0, end=args.end_noise).to(device)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x0 = batch.to(device)  # (batch, seq_len)
            b = x0.size(0)
            t = torch.randint(1, args.T + 1, (b,), device=device, dtype=torch.long)
            alpha_t = alphas[t].unsqueeze(1)  # (batch,1)

            # corrupt x0 -> xt via uniform replacement per token
            # (Меняем на случайные значения из random_tokens случайные токены из replace_mask)
            rand = torch.rand(x0.shape, device=device)
            replace_mask = rand >= alpha_t
            random_tokens = torch.randint(0, vocab_size, x0.shape, device=device, dtype=torch.long)
            xt = x0.clone()
            xt[replace_mask] = random_tokens[replace_mask]

            logits = model(xt, t)
            loss = criterion(logits.view(-1, vocab_size), x0.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            global_step += 1
            pbar.set_postfix({'loss': float(loss.item()), 'step': global_step})

            if global_step % args.save_every == 0:
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'tokenizer_path': args.tokenizer_path,
                    'args': vars(args),
                    'step': global_step
                }, os.path.join(args.out_dir, f'ckpt_step{global_step}.pt'))

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_path': args.tokenizer_path,
 'args': vars(args),
        'step': global_step
    }, os.path.join(args.out_dir, 'final.pt'))
    print("Training finished. Model saved.")

    print("Sampling examples:")
    for _ in range(10):
        text, ids = sample_from_model(model, tokenizer, args.seq_len, args.T, alphas, device)
        print(text)


# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--words_file', type=str, required=True, help='Path to file with one word per line')
    p.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')
    p.add_argument('--seq_len', type=int, default=8, help='Max token length per word (after BPE)')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--T', type=int, default=200, help='Number of diffusion steps')
    p.add_argument('--end_noise', type=float, default=0.9, help='Final noise level (prob of replacement at T)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--clip', type=float, default=1.0)

    # tokenizer options
    p.add_argument('--train_tokenizer', action='store_true', help='Train a BPE tokenizer from words_file')
    p.add_argument('--vocab_size', type=int, default=3000, help='If training tokenizer, target vocab size')
    p.add_argument('--tokenizer_path', type=str, default=None, help='Path to load/save BPE tokenizer (json)')

    # warm start
    p.add_argument('--resume_path', type=str, default=None, help='Checkpoint to resume from')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
