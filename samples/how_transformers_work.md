# How Transformers Work

## From text to tokens

A language model never sees letters or words directly. A tokenizer first splits the text into tokens: whole words, pieces of words, punctuation and spaces. Most modern tokenizers use byte-pair encoding (BPE), which starts from single characters and repeatedly merges the pairs that appear most often in the training text, so common words become one token and rare words are split into several. A word such as "unbelievable" might become "un", "believ" and "able". In English a token averages about four characters, or roughly three quarters of a word. Each token is then replaced by its ID, a number from a fixed vocabulary that usually holds between 30,000 and 200,000 entries.

## Token embeddings

Each token ID is used to look up a row in an embedding table: a list of numbers, a few hundred to several thousand long, that the model learns during training. Tokens used in similar ways end up with similar lists, so "cat" and "kitten" sit closer together than "cat" and "carburettor". Sentence embedding models build on the same idea for whole passages. The model used in this explorer, all-MiniLM-L6-v2, runs a small six-layer transformer over a chunk and averages its token vectors (mean pooling) into a single list of 384 numbers that stands for the meaning of the whole chunk.

## Adding word order

Attention on its own treats a sentence as a bag of tokens and has no idea which came first, so information about position has to be added. The original transformer added fixed patterns of sine and cosine waves to each token's embedding, a different pattern for each position. Many current models use rotary position embeddings (RoPE) instead, which rotate the query and key vectors by an angle that depends on the token's position, so attention scores reflect how far apart two tokens are.

## Self-attention: how tokens look at each other

Self-attention is the heart of the transformer. For every token, the model computes three vectors from its embedding: a query (what this token is looking for), a key (what this token offers) and a value (the information it passes on). To update one token, the model takes the dot product of that token's query with the key of every token in the text, divides the scores by the square root of the key length to keep them in a stable range, and applies softmax so the scores become weights that add up to 1. The token's new vector is the weighted sum of all the value vectors. In one line: Attention(Q, K, V) = softmax(QKᵀ / √d_k) V.

Take the sentence "The animal didn't cross the street because it was too tired." When the model updates the token "it", the query of "it" matches the key of "animal" strongly, so most of the weight goes there and the new vector for "it" carries information about the animal. Change "tired" to "wide" and the weight moves to "street".

## Multi-head attention

One set of attention weights can only capture one kind of relationship at a time, so transformers run several attention heads side by side, each with its own query, key and value projections. One head might follow grammar, linking verbs to their subjects, while another links pronouns to the nouns they refer to. The original transformer used 8 heads per layer. The outputs of all heads are joined together and projected back to the model's working size.

## The transformer block

A transformer is a stack of identical blocks. Each block applies multi-head attention, then a small feed-forward neural network to every token separately. Around each of these two parts sits a residual connection, which adds the part's input to its output so information and gradients flow easily through deep stacks, and layer normalisation, which keeps the numbers in a stable range. The original model had 6 blocks in its encoder and 6 in its decoder; large modern language models stack around a hundred.

## Encoders, decoders and decoder-only models

The original transformer was built for translation and had two halves: an encoder that reads the source sentence and a decoder that writes the translation. BERT keeps only the encoder. It reads text in both directions at once, which makes it good for classification, search and embeddings. GPT-style models keep only the decoder. They use a causal mask so each token can attend only to the tokens before it, and they are trained to predict the next token. Almost all chat models today are decoder-only.

## Generating text, one token at a time

At each step a decoder-only model outputs a probability for every token in its vocabulary. One token is chosen, added to the end of the text, and the whole process repeats until the model produces an end token or reaches a length limit. Temperature controls how the choice is made: a low temperature almost always picks the most likely token and gives predictable text, while a higher temperature gives more varied and creative text, with more risk of mistakes. Top-p, or nucleus, sampling keeps only the smallest set of tokens whose probabilities add up to p, such as 0.9, and samples from those.

## The context window

The context window is the largest number of tokens a model can attend to at once, counting both the prompt and the answer. Plain attention compares every token with every other token, so its cost grows with the square of the length: doubling the text makes attention about four times as expensive. GPT-3 had a context window of 2,048 tokens; recent models offer hundreds of thousands, and some more than a million. Even so, retrieval is still useful. Sending only the relevant passages is cheaper and faster, and models can overlook facts placed in the middle of very long inputs, an effect researchers call "lost in the middle".
