# A Short History of Artificial Intelligence

## Can machines think? (1950)

In 1950 Alan Turing published "Computing Machinery and Intelligence", which opens with the question "Can machines think?". Because "think" is hard to define, Turing replaced the question with a game he called the imitation game: a judge exchanges typed messages with a person and a machine, and the machine wins if the judge cannot reliably tell which is which. The idea is now known as the Turing test. Turing also predicted that machines would one day learn from experience, much as a child does, rather than having every rule written in by hand.

## The Dartmouth workshop (1956)

In 1955 John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon proposed a summer research project at Dartmouth College. Their proposal coined the term "artificial intelligence", and the workshop held in the summer of 1956 is usually counted as the founding of the field. One of the programs discussed there was the Logic Theorist by Allen Newell, Herbert Simon and Cliff Shaw, which proved theorems from Whitehead and Russell's Principia Mathematica. Early researchers were optimistic: several predicted that machines as intelligent as people were only a generation away.

## The perceptron, an early neural network (1958)

In 1958 the psychologist Frank Rosenblatt introduced the perceptron, a simple model of a neuron that learns by adjusting the weights on its inputs whenever it makes a mistake. Built as hardware, the Mark I Perceptron learned to recognise simple shapes. In 1969 Minsky and Seymour Papert published the book "Perceptrons", which showed that a single-layer perceptron cannot learn some simple functions, such as XOR. The book helped shift attention and funding away from neural networks for more than a decade.

## ELIZA, the first chatbot (1966)

Joseph Weizenbaum at MIT wrote ELIZA in 1966. Its best-known script, DOCTOR, imitated a psychotherapist by matching patterns in what the user typed and turning statements back into questions ("I feel sad" became "Why do you feel sad?"). ELIZA understood nothing, yet many users felt it did and confided in it. The tendency to read understanding into a program's fluent replies is still called the ELIZA effect.

## The first AI winter (1970s)

Promises had run far ahead of results. Programs that worked on small, tidy problems failed on real ones, because the number of possibilities to search grew explosively, a problem called combinatorial explosion. In 1973 the Lighthill report to the British government judged that AI had failed to meet its goals, and funding in the United Kingdom was cut sharply. In the United States, DARPA also reduced open-ended AI funding. This period of shrinking money and interest became known as the first AI winter.

## Expert systems and the second AI winter (1980s)

AI returned in the 1980s with expert systems: programs built from hundreds or thousands of if-then rules written down by interviewing human experts. XCON, used by Digital Equipment Corporation, configured computer orders and was reported to save the company millions of dollars a year. Japan launched its ambitious Fifth Generation Computer project. But expert systems were brittle outside their narrow area and expensive to keep up to date. When the market for specialised Lisp machines collapsed in 1987 and many projects missed their goals, funding dried up again. This second AI winter lasted into the early 1990s.

## Backpropagation brings neural networks back (1986)

In 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams published a paper that popularised backpropagation, a way to train networks with several layers by passing the error backwards through the network and adjusting every weight a little. Multi-layer networks could learn functions such as XOR that single perceptrons could not. In the 1990s Yann LeCun's convolutional neural networks read handwritten digits, and banks used them to read the amounts on cheques.

## Deep Blue beats Kasparov (1997)

In May 1997 IBM's Deep Blue defeated the world chess champion Garry Kasparov in a six-game match, 3.5 to 2.5. Deep Blue did not learn the way modern systems do. It relied on special-purpose hardware that searched around 200 million chess positions per second, guided by an evaluation function tuned by chess experts. It showed that raw search could beat a human champion in a well-defined game.

## The deep learning breakthrough (2012)

In 2012 a convolutional neural network called AlexNet, built by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton, won the ImageNet image recognition challenge with a top-5 error rate of about 15 percent, against about 26 percent for the runner-up. Three things came together: very large labelled datasets, graphics processing units (GPUs) that could train big networks quickly, and better training techniques. Within a few years deep learning led speech recognition, translation and image recognition.

## AlphaGo (2016)

Go has far more possible positions than chess, so brute-force search alone could not master it. In March 2016 DeepMind's AlphaGo defeated Lee Sedol, one of the strongest Go players in the world, by four games to one in Seoul. AlphaGo combined deep neural networks, which judged positions and suggested promising moves, with Monte Carlo tree search. It learned first from human games and then by playing millions of games against itself. Its move 37 in the second game surprised professional players, who first thought it was a mistake.

## The transformer (2017)

In 2017 researchers at Google published "Attention Is All You Need", which introduced the transformer architecture. Earlier language models read text one word at a time with recurrent networks. The transformer replaced that with attention, which lets every word look at every other word at once, so training could run in parallel on GPUs. The transformer became the basis of BERT (2018) and of the GPT family of models.

## Large language models and ChatGPT (2018 to 2022)

OpenAI's GPT (2018), GPT-2 (2019, 1.5 billion parameters) and GPT-3 (2020, 175 billion parameters) showed that simply making a transformer bigger and training it on more text kept improving it. GPT-3 could often do a new task from a few examples written in the prompt, without any retraining, which is called few-shot learning. On 30 November 2022 OpenAI released ChatGPT, a chat model fine-tuned with reinforcement learning from human feedback. It was estimated to have reached 100 million users in about two months, faster than any consumer application before it.

## Today: retrieval, tools and many kinds of input

Current models read images, audio and video as well as text. Retrieval-Augmented Generation (RAG) connects a model to documents it was never trained on, such as a company's own files. Agents let a model call tools, search the web or run code to complete a task in several steps. The open problems are familiar ones: making answers reliable, stopping models from stating false things confidently, lowering cost, and keeping powerful systems safe.
