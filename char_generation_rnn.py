"""
Character-Level Text Generation RNN (PyTorch)
-----------------------------------------------
Task: given a sequence of characters, predict the *next* character.
Trained autoregressively, then used to generate new text one char at a time.

Why char-level (and not word-level) for a first RNN project?
- Tiny, fixed vocabulary (just the unique characters in the text — usually <100),
  so no embedding-table-size or out-of-vocabulary issues to worry about yet.
- Every design choice you make here (hidden state shape, batching, loss,
  sampling) is the same one you'll use in a word-level or seq2seq model later —
  this task just removes the tokenization complexity so the RNN mechanics
  are the only thing in focus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. DATA: load text, build char <-> index vocabulary
# ---------------------------------------------------------------------------
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))          # every unique character in the corpus
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Corpus length: {len(text):,} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

# Encode the ENTIRE corpus as one long tensor of integer indices.
# This is the only encoding step — the RNN never sees raw text, only indices.
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)


# ---------------------------------------------------------------------------
# 2. BATCHING: sample random (input, target) sequence pairs
# ---------------------------------------------------------------------------
# For next-char prediction, the target sequence is just the input sequence
# shifted right by one position. E.g. if input = "hell", target = "ello".
# At every position t, the model sees data[t] and is trained to predict data[t+1].

SEQ_LEN = 100      # how many characters the RNN unrolls over per training step
BATCH_SIZE = 64


def get_batch():
    # pick BATCH_SIZE random starting points in the corpus
    max_start = len(data) - SEQ_LEN - 1
    starts = torch.randint(0, max_start, (BATCH_SIZE,))
    x = torch.stack([data[s:s + SEQ_LEN] for s in starts])
    y = torch.stack([data[s + 1:s + SEQ_LEN + 1] for s in starts])
    return x, y  # both shape: (BATCH_SIZE, SEQ_LEN)


# ---------------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=1):
        super().__init__()
        # Embedding: maps each character index -> a learned dense vector.
        # This replaces one-hot vectors; the network learns which characters
        # behave similarly (e.g. vowels) instead of treating every char index
        # as equally unrelated to every other.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # nn.RNN implements exactly the recurrence h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
        # for every time step, with the SAME W_xh/W_hh reused at each step —
        # this is the parameter-sharing property that let RNNs handle variable-length
        # sequences, discussed when we covered why ANNs weren't enough.
        # batch_first=True means input/output tensors are shaped (batch, seq, feature)
        # instead of PyTorch's default (seq, batch, feature).
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer maps each time step's hidden state -> a score per vocab character.
        # Applied independently at every position (it doesn't know about time at all —
        # all the sequence modeling already happened inside the RNN).
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)  — integer indices
        x = self.embedding(x)                  # -> (batch, seq_len, embed_dim)
        out, hidden = self.rnn(x, hidden)       # out: (batch, seq_len, hidden_dim)
        #   hidden: (num_layers, batch, hidden_dim) — the LAST time step's hidden
        #   state, which is what you feed back in to continue generation.
        out = self.fc(out)                      # -> (batch, seq_len, vocab_size)
        return out, hidden


# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharRNN(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 2000
PRINT_EVERY = 200

model.train()
for epoch in range(1, EPOCHS + 1):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, _ = model(x)  # logits: (batch, seq_len, vocab_size)

    # CrossEntropyLoss expects (N, C) vs (N,), so we flatten the batch and
    # time dimensions together — every (sample, time-step) pair is treated
    # as one independent classification instance over the vocabulary.
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    loss.backward()
    optimizer.step()

    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 5. GENERATION (autoregressive sampling)
# ---------------------------------------------------------------------------
def generate(model, seed="T", length=400, temperature=0.8):
    """
    Feed one character at a time, carry the hidden state forward, and sample
    the next character from the model's predicted distribution — then feed
    that SAMPLED character back in as the next input. This is why generation
    is sequential/slow: each step depends on the output of the previous one,
    same as the recurrence itself.
    """
    model.eval()
    hidden = None
    generated = seed

    # "warm up" the hidden state on the seed text first
    input_idx = torch.tensor([[char_to_idx[c] for c in seed]], device=device)
    with torch.no_grad():
        logits, hidden = model(input_idx, hidden)

    last_char_idx = torch.tensor([[char_to_idx[seed[-1]]]], device=device)

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(last_char_idx, hidden)
            # logits shape: (1, 1, vocab_size) -> take the single time step
            probs = F.softmax(logits[0, -1] / temperature, dim=0)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated += idx_to_char[next_idx]
            last_char_idx = torch.tensor([[next_idx]], device=device)

    return generated


print("\n--- Generated sample ---\n")
print(generate(model, seed="ROMEO:", length=400, temperature=0.8))

# Save the trained weights so you don't have to retrain to generate again later
torch.save(model.state_dict(), "char_rnn.pt")
print("\nModel saved to char_rnn.pt")
