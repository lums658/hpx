# Phylanx Algorithm Ports to HPXPy

This document describes three advanced machine learning algorithms ported from Phylanx to HPXPy, demonstrating different distributed computing patterns.

## Ported Algorithms

### 1. Alternating Least Squares (ALS) - `als_recommender.ipynb`

**Application:** Recommendation systems (movies, products, music)

**Algorithm Type:** Matrix factorization for collaborative filtering

**Key Features:**
- Factorizes sparse rating matrix into user and item latent factors
- Confidence weighting for implicit feedback
- Alternating optimization: fix items → optimize users, then fix users → optimize items

**Distributed Pattern:** Block alternation with parameter broadcasting
- Phase 1: All users updated in parallel (broadcast item factors)
- Phase 2: All items updated in parallel (broadcast user factors)
- Perfect parallelism within each phase
- Communication: O(factors × (users + items)) per iteration

**Real-World Scale:**
- Netflix: 480K users, 18K movies
- Spotify: 320M users, 70M tracks
- Amazon: 300M users, 350M products

**Educational Value:**
- Demonstrates parameter server pattern
- Shows confidence-weighted learning
- Least squares solves in distributed setting
- Near-linear scaling with embarrassingly parallel updates

---

### 2. Neural Network with Backpropagation - `neural_network.ipynb`

**Application:** Classification, regression, function approximation

**Algorithm Type:** Single-hidden-layer feedforward network

**Key Features:**
- Sigmoid activation functions
- Forward propagation through hidden layer
- Backpropagation via chain rule
- Gradient descent optimization

**Distributed Pattern:** Data parallelism (foundation of modern deep learning)
- Distribute mini-batches across nodes
- Each node: forward pass → backward pass → compute gradients
- All-reduce to aggregate gradients
- Synchronous weight updates

**Real-World Scale:**
- ImageNet training: 1.3M images, 8× GPUs, ~1 hour
- BERT: 3.3B words, 64× TPUs, 4 days
- GPT-3: 175B parameters, 10K GPUs, 2 weeks, $4.6M

**Educational Value:**
- Foundation of deep learning frameworks (PyTorch, TensorFlow)
- Shows gradient computation via chain rule
- Demonstrates synchronous SGD pattern
- Communication/computation trade-off analysis

---

### 3. Latent Dirichlet Allocation (LDA) - `lda_topic_modeling.ipynb`

**Application:** Topic modeling for document collections

**Algorithm Type:** Probabilistic inference via Gibbs sampling (MCMC)

**Key Features:**
- Discovers hidden topics in text corpora
- Bayesian inference with Dirichlet priors
- Gibbs sampling for posterior distribution
- Rejection sampling technique

**Distributed Pattern:** Document parallelism with sparse communication
- Partition documents across nodes
- Local Gibbs sampling (independent per document)
- Synchronize word-topic counts (sparse matrix)
- Document-topic counts stay local (no communication!)

**Real-World Scale:**
- PubMed: 30M biomedical abstracts, 100 nodes, ~4 hours
- Wikipedia: 6.4M articles, 1000 nodes, ~8 hours
- Twitter: 500M tweets/day, online learning

**Educational Value:**
- Unique algorithm class: probabilistic inference
- MCMC sampling methods
- Sparse communication optimization (90-99% zeros)
- Asynchronous updates work well (tolerates staleness)
- Document independence enables perfect parallelism

---

## Comparison of Distributed Patterns

| Algorithm | Pattern | Parallelism | Communication | Key Bottleneck |
|-----------|---------|-------------|---------------|----------------|
| **ALS** | Block alternation | Embarrassingly parallel within phase | O(factors × entities) | Matrix inversion |
| **Neural Network** | Data parallelism | Parallel per batch | O(model_size) | Gradient sync |
| **LDA** | Document parallelism | Embarrassingly parallel | O(vocab × topics), sparse | Sampling speed |

### Communication Efficiency Ranking

1. **LDA (Best):** Sparse word-topic matrix, documents stay local
2. **ALS:** Only factor matrices broadcast, not rating matrix
3. **Neural Network:** Full model gradients need aggregation

### Scaling Characteristics

- **ALS:** Linear speedup, limited only by synchronization barriers
- **Neural Network:** Linear speedup until communication dominates
- **LDA:** Super-linear possible with sparse optimization + async updates

---

## Distributed Computing Lessons

### Pattern 1: Embarrassingly Parallel (ALS, LDA)
When updates are independent, you get perfect linear scaling.

### Pattern 2: Synchronous All-Reduce (Neural Network)
Communication cost increases with model size. Critical ratio: compute_time / comm_time > 10.

### Pattern 3: Sparse Communication (LDA)
Exploit sparsity to reduce communication by 10-100×.

### Pattern 4: Local vs Global State
- **Global:** Must synchronize (word-topic in LDA, factor matrices in ALS)
- **Local:** No communication (document-topic in LDA, gradients before reduction in NN)

---

## Implementation Notes

### From Phylanx to HPXPy

**What Changed:**
1. Removed `@Phylanx` decorator (not needed in HPXPy)
2. Used NumPy directly instead of Phylanx primitives
3. Made parallelism explicit in comments (HPXPy uses explicit execution policies)
4. Added extensive documentation for distributed patterns

**What Stayed the Same:**
1. Core algorithm logic (identical mathematical operations)
2. Data structures (NumPy arrays)
3. Control flow (loops, conditionals)

**HPXPy Advantages:**
- Direct NumPy compatibility (zero-copy when C-contiguous)
- Explicit execution control (sequential, parallel, parallel+SIMD)
- No compilation overhead (no AST transformation)
- Simpler mental model (direct function calls)

**Phylanx Advantages:**
- Automatic optimization (compiler-driven)
- Pattern matching for distributed operations
- PhySL domain-specific language
- Built-in support for distributed primitives

---

## How to Run

All three notebooks are self-contained and can be run directly:

```bash
cd /Users/lums/LSU/hpx/python/examples/

# Run in Jupyter
jupyter notebook als_recommender.ipynb
jupyter notebook neural_network.ipynb
jupyter notebook lda_topic_modeling.ipynb
```

Each notebook includes:
- ✅ Complete working implementation
- ✅ Synthetic data generation (no external datasets required)
- ✅ Performance timing
- ✅ Result visualization
- ✅ Detailed comments explaining distributed patterns
- ✅ Scaling analysis with real-world examples

---

## Future Extensions

### Short Term
1. Add HPXPy parallel execution policies (`policy="par"`) to key operations
2. Implement multi-locality versions using HPXPy collectives
3. Add GPU acceleration using `hpx.gpu` module
4. Create distributed versions with `hpx.distributed_*` functions

### Medium Term
1. **ALS:** Implement implicit feedback variant, add regularization options
2. **Neural Network:** Add more layers, different activation functions, mini-batch support
3. **LDA:** Implement online LDA for streaming data, hierarchical topics

### Long Term
1. Create benchmark suite comparing with NumPy, scikit-learn, Spark MLlib
2. Add visualization tools (t-SNE embeddings, topic word clouds)
3. Implement advanced variants (FastALS, Asynchronous SGD, Async LDA)
4. Port additional Phylanx algorithms (Random Forest, t-SNE, QR decomposition)

---

## References

### ALS
- Hu, Koren, Volinsky. "Collaborative Filtering for Implicit Feedback Datasets" ICDM 2008
- Original Phylanx: `/Users/lums/LSU/phylanx/examples/algorithms/als/`

### Neural Networks
- Analytics Vidhya Tutorial (Sunil Ray)
- Original Phylanx: `/Users/lums/LSU/phylanx/examples/algorithms/nn/`

### LDA
- Newman et al. "Distributed Algorithms for Topic Models" JMLR 2009
- Newman et al. "Distributed Inference for Latent Dirichlet Allocation" NIPS 2007
- Original Phylanx: `/Users/lums/LSU/phylanx/examples/algorithms/lda/`

---

## Contributing

These examples demonstrate three fundamental distributed computing patterns:
1. **Block alternation** (ALS)
2. **Data parallelism** (Neural Network)
3. **Document/entity parallelism** (LDA)

Understanding these patterns enables implementing most distributed ML algorithms. Consider contributing:
- Additional Phylanx ports (t-SNE, Random Forest, QR decomposition)
- Performance optimizations
- Multi-locality implementations
- GPU acceleration
- Real-world dataset examples

---

**Status:** ✅ All three algorithms successfully ported and documented
**Date:** January 2026
**Ported by:** Claude Code from Phylanx examples
