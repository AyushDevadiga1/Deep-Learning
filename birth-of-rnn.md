Here is a breakdown of each landmark experiment and paradigm from the 1980s, highlighting the core insight and directly emphasizing the key examples for each.These are various discoveries from different researcher from Old Days to the birth of RNN.

---

### 1. Symbolic AI & Formal Rule Grammars

* **Meaningful Insight:** Language structure was modeled purely through human-engineered context-free grammar rules without learning from data. The machine followed rigid parsing logic, assuming human language could be fully captured by explicit syntactic production rules.
* **Key Examples:**
1. **Success Example:** Parsing simple, standard sentences like `"The cat chased the mouse"` by splitting the string into exact syntactic branches ($\text{NounPhrase} + \text{VerbPhrase}$).
2. **Failure Example (Structural Ambiguity):** In `"I saw the man with the telescope"`, the system cannot determine if *"with the telescope"* attaches to `"saw"` or `"man"`, producing competing valid trees with no data-driven method to decide which is more likely.
3. **Failure Example (Noise & Conversational Syntax):** Inputting human speech like `"Um, so the cat, like, totally ran"` causes total parse failure because the explicit grammar lacks rules for conversational fillers.



---

### 2. $n$-Gram Language Models & Hidden Markov Models (HMMs)

* **Meaningful Insight:** Shifted the paradigm to statistical learning from text corpora, but introduced the **Markov assumption**—limiting context to a small, fixed window of $n-1$ recent tokens to prevent exponential parameter explosion.
* **Key Examples:**
1. **Success Example:** Predicting short-range word patterns like predicting `"York"` given `"New"`.
2. **Failure Example (Distant Agreement):** Predicting the verb in `"The chef who cooked the extraordinary seven-course meals [was/were] praised."`
* A **3-gram model** only evaluates the preceding two tokens: `("course", "meals")`.
* Because `"meals"` is plural, the model incorrectly assigns a higher probability to **"were"**, missing the true singular subject (**"chef"**) located 7 steps back outside its lookback window.





---

### 3. Time-Delay Neural Networks (TDNN — Waibel et al., 1989)

* **Meaningful Insight:** Adapted feedforward neural networks to continuous time-series data (primarily speech) by feeding a fixed temporal sliding window of inputs simultaneously with shared weights.
* **Key Examples:**
1. **Success Example:** Local phoneme recognition, such as identifying the 30–50 millisecond acoustic transition of the stop consonant **/p/** in `"pat"`.
2. **Failure Example (Context Window Truncation):** Distinguishing vowel durations in phoneme pairs like *"bit"* vs. *"beat"*, where co-articulation effects stretch across 15–20 audio frames (150–200ms). If the TDNN window is set to 5 frames, any acoustic cue occurring 6 or more frames back is mathematically invisible to the model.



---

### 4. Hopfield Networks (Hopfield, 1982)

* **Meaningful Insight:** Demonstrated that feedback loops in a neural network could settle into stable energy minima, providing theoretical proof that recurrence could be harnessed as a stable associative memory rather than devolving into chaotic feedback noise.
* **Key Examples:**
1. **Success Example (Pattern Reconstruction):** Storing binary images (such as handwritten digits) as target states. When presented with a corrupted or pixelated version of a digit, the network iteratively updates its recurrent states until it converges back to the clean, original stored digit.



---

### 5. Jordan Recurrent Networks (Jordan, 1986)

* **Meaningful Insight:** Introduced task-driven temporal behavior by feeding the network's **previous output** back into its input layer at the next time step, allowing output history to influence future decisions.
* **Key Examples:**
1. **Success Example (Motor Trajectories):** Generating smooth sequence trajectories for robotics, where outputting joint angle position $\theta_1$ at step 1 feeds back to help determine the next position $\theta_2$ at step 2.
2. **Failure Example (Hidden Information Bottleneck):** In complex NLP tasks, if the network outputs a discrete classification tag like `"Noun"`, feeding back only that single label discards all rich intermediate representations calculated by the internal hidden layers during execution.



---

### 6. Elman Recurrent Neural Networks (Elman, 1990)

* **Meaningful Insight:** Fed the **hidden layer's own previous activation** back into itself. This created an implicit, continuous memory state vector ($h_t$) that recursively updates at every step:

$$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$$

This enabled the network to maintain compressed historical representations without fixed windows or hand-coded grammars.

* **Key Examples:**
1. **Success Example (Long-Distance Tracking):** In `"The boy who chased the cats [was/were]..."`, processing `"boy"` at step 2 updates hidden state $h_2$ to encode `[singular_subject]`. As intermediate tokens pass, $h_t$ updates continuously while preserving a residual trace of $h_2$. At step 7, $h_6$ retains enough historical context to predict the singular verb **"was"**, bypassing the fixed-window limitation that broke $n$-gram and TDNN models.