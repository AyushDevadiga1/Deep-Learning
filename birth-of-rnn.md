
---

### Why did we need something *other* than what we already had?

A CNN is very good at exploiting **spatial locality** — nearby pixels are related, and a convolution kernel slides over that space, sharing weights across positions. That's a strong assumption baked into the architecture.

Now ask: what assumption does a plain feedforward net (MLP) or a CNN make about its *input as a whole*?

It assumes the input is a **fixed-size, static blob**. Feed an image → get an output. Feed the *next* image → completely independent computation. There is no concept of "this input came after that one" or "the meaning of this element depends on what came before it in a sequence."

That's fine for classifying a photo. It breaks immediately for:
- A sentence, where "bank" means something different after "river" than after "money"
- Speech, where the current sound depends on the sounds before it
- A stock price series, where today's value is contextualized by yesterday's

Three concrete failure modes of using an MLP/CNN for this:
1. **No memory** — no mechanism carries information from step *t-1* to step *t*.
2. **Fixed input size** — an MLP needs a predetermined number of inputs; a sentence isn't a fixed length.
3. **No parameter sharing across time** — even if you hacked a fixed-length window into an MLP, you'd need separate weights for "word in position 1" vs "word in position 5," which doesn't generalize — the network never learns "the concept of the previous word," only "whatever happened to be in slot 3."

So the actual design goal that had to be invented was: **a network with an internal state that updates as it consumes a sequence, one element at a time, using the *same* weights at every step.**

That one sentence is the whole reason RNNs exist. Everything else is engineering how to make that state actually work well.

---

### Where did this idea actually come from?

This wasn't invented once — it came in three steps, each solving a piece of the puzzle.

**1. Hopfield Networks (John Hopfield, 1982)**
Not built for sequence prediction at all — built as **associative memory**. Every neuron connects to every other neuron (fully recurrent), and the network settles into a stable "energy minimum" state given a partial or noisy input — that's how it "recalls" a stored pattern. The important contribution here wasn't the task, it was proving that **recurrent connections (loops in the graph) can be trained and can hold meaningful state** — before this, "loops" in a network were mostly seen as a stability nightmare, not a memory mechanism.

**2. Jordan Networks (Michael Jordan, 1986)**
Applied recurrence to actual sequential output — originally for motor sequences (robot control). The trick: feed the network's **own previous output** back in as an extra input for the next step. Simple, but it meant the network's next decision could depend on what it had *just produced* — a first notion of temporal dependency in a task-driven (not just memory-recall) setting.

**3. Elman Networks (Jeffrey Elman, 1990) — "Finding Structure in Time"**
The real ancestor of what you'll build. Instead of feeding back the *output*, Elman fed back the **hidden layer's own previous activation** as an additional input at the next time step. This is the key shift: the hidden state itself becomes a compressed, learned summary of everything seen so far — not just "the last output," but "the last internal representation." Elman used this on language tasks and showed the network could implicitly learn structure like word categories and simple grammar, purely from sequence prediction.

---
