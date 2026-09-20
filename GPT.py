import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, mask):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float("-inf"), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss and (logits, loss) or (logits, None)


model = TinyGPT(vocab_size=50257, n_embd=384, n_head=6, n_layer=6, block_size=256)
state_dict = torch.load(
    r"C:\Users\Dr. Lone\Desktop\Final_weights\results\tinygpt_final_weights.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()
print("Loaded successfully")


tokenizer = AutoTokenizer.from_pretrained("gpt2") 

def generate(model, tokenizer, prompt, max_new_tokens=100):
    idx = torch.tensor([tokenizer.encode(prompt)])
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -256:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return tokenizer.decode(idx[0].tolist())

print(generate(model, tokenizer, "Once upon a time, there was a"))