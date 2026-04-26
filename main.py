import torch
import torch.nn as nn
import math
import torch.nn.functional as f
from dataclasses import dataclass
import tiktoken

class PositionalEncoding(nn.Module):  #unlimited length
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * (torch.arange(0, d_model, 2) / d_model))  #div = 10000^(2i/d)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiAttentionHead(nn.Module):
    def __init__(self, d_model, head_counts=1, drop_out=0.1, cache=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = self.d_model // head_counts
        self.scale = self.d_k ** 0.5
        self.head_counts = head_counts
        self.Q_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.K_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.V_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.O_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attention_dropout = nn.Dropout(drop_out)
        self.proj_dropout = nn.Dropout(drop_out)
        self.cache = cache

    def head_splits(self, data, ):
        batch, seq, dim = data.size()
        return data.view(batch, seq, self.head_counts, dim // self.head_counts).transpose(1, 2)

    def forward(self, x, last_layer=None, ):
        batch, seq, dim = x.size()

        q = self.Q_weight(x)
        k = self.K_weight(x)
        v = self.V_weight(x)

        q = self.head_splits(q)
        k = self.head_splits(k)
        v = self.head_splits(v)

        if last_layer is not None:
            past_k, past_v = last_layer
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k,v) if self.cache else None
        score = torch.matmul(q, k.transpose(-2, -1))

        if last_layer is None:
            mask = torch.tril(torch.ones(seq, seq, device=x.device)).view(1, 1, seq, seq)

            score = score.masked_fill(mask == 0, float('-inf'))

        attention = f.softmax(score / self.scale, dim=-1)
        attention = self.attention_dropout(attention)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        out = self.O_weight(out)
        out = self.proj_dropout(out)

        return out, present

class MLP(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.dff = 4 * d_model #subjective to change
        self.fc1 = nn.Linear(d_model, self.dff, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.dff, d_model, bias=False)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, head_counts=1, p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiAttentionHead(d_model, head_counts, p, True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, p)

    def forward(self, x, last_layer=None):
        atten, present = self.attention(self.ln1(x), last_layer)
        x = x + atten
        x = x + self.mlp(self.ln2(x))
        return x, present

class Embedding(nn.Module):  #learned embedding
    def __init__(self, config, d_model):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, d_model)
        self.wpe = nn.Embedding(config.block_size, d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        batch, seq_len = idx.size()

        tok_emb = self.wte(idx)

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_emb + pos_emb)
        return x

@dataclass
class GPTConfig:
    block_size = 64
    vocab_size = 50257
    n_layer = 2
    n_head = 4
    n_embed = 128
    dropout = 0.0

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config, config.n_embed)
        self.blocks = nn.ModuleList([Block(config.n_embed, config.n_head, config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.head.weight = self.embedding.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, past_key_values=None):
        x = self.embedding(idx)
        presents = []
        for i, block in enumerate(self.blocks):
            last_layer = past_key_values[i] if past_key_values else None
            x, present = block(x, last_layer)
            presents.append(present)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, presents

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, block_size=1024):
    """
    Take a conditioning sequence of indices idx (B,T) and complete it
    to max_new_tokens length. model is expected to return logits only.
    Args:
        model: your GPT model
        idx: (B, T) tensor of indices in current context
        max_new_tokens: how many tokens to generate
        temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random
        top_k: if set, only sample from top k probabilities
        block_size: context length of your model
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it to block_size
        idx_cond = idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _, _ = model(idx_cond) # (B, T, vocab_size)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature # (B, vocab_size)
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to probabilities
        probs = torch.softmax(logits, dim=-1) # (B, vocab_size)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
if __name__ == "__main__":
    config = GPTConfig()

    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2,16))
    target = torch.randint(0, config.vocab_size, (2,16))
    logits, loss, _ = model(x, target)
    #print(logits.std())
    #print(loss.item())

    enc = tiktoken.get_encoding("gpt2")



    file = open("pharaoh.txt", "r")
    data = file.read()
    file.close()

    ids = enc.encode(data)
    idx = torch.tensor(ids, dtype=torch.long)

    # ===== BATCHING =====
    def get_batch(data, block_size, batch_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y



    # ===== TRAINING =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    model.train()

    max_steps = 1000
    batch_size = 1

    for step in range(max_steps):
        # 1. Get a batch of data
        X, Y = get_batch(idx, config.block_size, batch_size)

        # 2. Forward pass: compute loss
        logits, loss, present= model(X, Y)


        # 3. Backward pass: compute gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevents exploding gradients

        # 4. Update weights
        optimizer.step()

        # 5. Log progress every 10 steps
        if step % 10 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

        # 6. Save checkpoint every 200 steps
        if step % 200 == 0 and step > 0:
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'step': step,
            }, f'pharaoh_cpu_{step}.pt')
            print(f"Saved checkpoint at step {step}")





