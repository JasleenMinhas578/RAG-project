# How Language Models Are Trained, and Why They Need RAG

## Pretraining: predicting the next token

A large language model starts as a transformer with billions of random numbers, called parameters. During pretraining it reads an enormous amount of text, from web pages, books, articles and code, often trillions of tokens in total. Its only task is to predict the next token. Each time it guesses wrong, a loss function called cross-entropy measures how wrong it was, backpropagation works out how each parameter contributed to the error, and gradient descent nudges every parameter slightly in the direction that would have made the right token more likely. Pretraining a frontier model runs on thousands of GPUs or TPUs for weeks or months.

## What pretraining produces

The result is a base model. It has absorbed grammar, a great many facts, some reasoning ability and the style of every kind of writing it read, but it only continues text. Ask a base model a question and it may answer, or it may continue with more questions, because that is also a likely continuation. Its knowledge is also frozen at its training cutoff: it knows nothing that happened, or was written, after its training data was collected.

## Supervised fine-tuning

To turn a base model into an assistant, it is trained further on example conversations written or checked by people: an instruction followed by a good response. This step, called supervised fine-tuning or instruction tuning, teaches the model the format of a conversation, to follow instructions and to answer helpfully. It needs far less data than pretraining, typically tens of thousands of examples rather than trillions of tokens.

## RLHF: reinforcement learning from human feedback

Writing perfect answers is slow, but comparing answers is quick. In reinforcement learning from human feedback (RLHF), the model writes several responses to the same prompt and people pick the one they prefer. A separate reward model is trained on those preferences, so it learns to predict which answer a person would choose. The language model is then optimised, often with an algorithm called proximal policy optimisation (PPO), to write responses that the reward model scores highly, while a penalty stops it from drifting too far from the fine-tuned model. Direct preference optimisation (DPO) reaches a similar result without training a separate reward model. In Constitutional AI and other forms of reinforcement learning from AI feedback (RLAIF), an AI model guided by a written set of principles provides some of the feedback instead of people.

## Why models hallucinate

A hallucination is an answer that sounds fluent and confident but is false or not supported by any source. Models hallucinate for several reasons. They are trained to produce plausible text, not to check facts. Rare facts appear only a few times in the training data and are learned poorly. Anything after the training cutoff is unknown. And a model has never seen private documents such as a company's handbook or your own notes, so when asked about them it can only guess. Invented citations, with real-looking authors and titles that do not exist, are a well-known example.

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation was described in a 2020 paper by Patrick Lewis and colleagues at Facebook AI Research, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". The idea is simple: before the model answers, find the passages most relevant to the question and place them in the prompt, with an instruction to answer from them. A typical pipeline splits documents into chunks, turns each chunk into an embedding, stores the embeddings in a vector index, embeds the question with the same model, retrieves the closest chunks, builds the prompt and generates the answer. This is exactly the pipeline this explorer shows step by step.

RAG reduces hallucination because the model no longer has to recall facts from memory: the facts are in front of it. It also lets a model use private or up-to-date information without any retraining, and because each passage is labelled, answers can cite their sources so a reader can check them. Its weakness is retrieval itself. If the right chunk is not retrieved, the model either says it does not know or, worse, falls back on guessing, so a RAG system is only as good as its search.

## Vector databases and similarity search

A vector database stores embeddings and answers one question quickly: which stored vectors are closest to this new one? Exact search compares the question with every stored vector, which is what FAISS's IndexFlatL2 does in this explorer and is fast enough for thousands of chunks. For millions of vectors, approximate nearest neighbour indexes trade a little accuracy for a large speed-up: HNSW builds a graph that can be walked towards the nearest points, and IVF clusters the vectors and searches only the most promising clusters. Closeness is usually measured with cosine similarity, which compares the direction of two vectors and ignores their length.

## Keyword search still matters

Embeddings capture meaning, so they find a passage even when it uses different words from the question. But they can miss exact strings that matter: names, product codes, error numbers and acronyms such as RLHF. Keyword methods such as TF-IDF and BM25 score passages by the exact words they share with the question, giving extra weight to rare words. Many production systems run both kinds of search and merge the two ranked lists, for example with reciprocal rank fusion. This is called hybrid search.

## Fine-tuning or RAG?

Fine-tuning and RAG solve different problems. Fine-tuning changes how a model behaves: its tone, its output format, or its skill at a narrow task. RAG changes what the model knows at the moment it answers. Facts that change often, or that are private, belong in retrieval, because updating a document is far cheaper than retraining a model. Many real systems use both: a fine-tuned model that is good at answering from retrieved passages.

## Evaluating a RAG system

A RAG system can fail in two places, so it is measured in two parts. Retrieval metrics ask whether the right chunks came back, for example recall at k: the share of questions for which a relevant chunk appears in the top k results. Generation metrics ask whether the answer is faithful to the retrieved text, relevant to the question and correct. Because people cannot read thousands of answers, teams often use another language model as a judge to grade them. LLM judges tend to be lenient, so their scores are best used to spot problems and are regularly checked against human ratings.
