# UT Austin Class Projects

## Natural Language Processing (Fall 2025)

### Sentiment Classification & Fairness
Implemented multiclass logistic regression with sparse bag-of-words and dense word embeddings. Trained feedforward neural networks on embeddings. Audited model behavior on identity-linked data slices and analyzed fairness/bias mitigation strategies.

### Word Embeddings & Language Modeling
Worked with Word2Vec and GloVe embeddings to measure semantic similarity and solve analogy tasks. Built n-gram and RNN-based language models, evaluated with perplexity. Implemented dependency parsing to show how syntax resolves ambiguity.

### Attention Mechanisms
Implemented scaled dot-product attention, multi-head self-attention, and positional encodings from scratch. Ran ablation studies on positional information and attention dropout. Visualized attention heatmaps to understand masking and structure effects on language modeling.

### Transformer Architecture
Built three transformer variants: encoder-only, decoder-only, and encoder-decoder. Explored how architectural choices affect modeling behavior. Implemented RLHF and DPO alignment loops to tune models on synthetic preference data.

### Capstone Project
Trained a compact decoder-only transformer on WikiText-2 or similar datasets. Ran comparative experiments on architectural choices (positional encodings, dropout, vocabulary size). Fine-tuned pretrained models (DistilGPT-2, T5-small) on downstream tasks. Visualized attention and saliency maps to interpret model behavior through lens of course concepts.

## Artificial Intelligence Honors (Fall 2024)

### Pathfinding & Search
Implemented DFS, BFS, uniform cost search, and A* algorithms to navigate mazes. Solved complex state-space problems (corners, eating all dots) by combining search algorithms with effective heuristics.

### Adversarial Game AI
Built minimax and alpha-beta pruning for competitive gameplay. Learned that search depth paired with strong evaluation functions outperforms naive approaches. Implemented expectimax for probabilistic opponent modeling.

### Reinforcement Learning
Developed value iteration for offline policy computation and Q-learning for online learning. Scaled from exact methods to approximate Q-learning with feature-based representations on large state spaces.

### Multi-Agent Systems
Designed offensive and defensive strategies for Capture the Flag competition. Integrated search, evaluation functions, and learned features under strict computational constraints. Emphasized iterative design and empirical testing.

### Probabilistic Inference
Implemented exact Bayesian inference and particle filtering to reason about hidden ghost positions from noisy sensor data. Built joint particle filters for correlated multi-agent behavior with Bayesian networks.

### Supervised Learning & Classification
Trained perceptron classifiers for digit recognition and behavior cloning. Learned that hand-engineered features dramatically outperform raw input representations in classification performance.

## Reinforcement Learning Graduate Class (Spring 2024)

### Multiarm Bandits
Created stationary and nonstationary multiarm bandits, balancing exploration and exploitation.

### Monte Carlo and TD Learning
Created Monte Carlo and n-step bootstrapping agents.

### Function Approximation
Created value functions with tile coding and neural network function approximation.

### Function Approximation with Policies
Created SARSA and REINFORCE agent using neural networks.

### Final Project
[View project repository](https://github.com/JustinSasek/RL-Final-Project)

## Computer Graphics Honors (Spring 2024)

See [course website](https://www.cs.utexas.edu/~graphics/s24/cs354h/) for full project specifications.

### Ray Tracer
Created Whitted-style ray tracer and stochastic path tracer in C++. Added support for reflections, refractions, depth of field, texture mapping, and skyboxes. Implemented RISTER-style importance sampling to improve low-light performance.

### Menger Sponge
Wrote GPU shaders to accelerate the rendering of a Menger Sponge fractal.

### Minecraft
Created procedurally generated infinite Minecraft world with chunk loading, biomes, and custom Perlin-noise texture shaders.

### Animation
Created animation software to manipulate trimeshes with bones. Implemented keyframe system to track keyframes and playback animation with quaternion interpolation.
