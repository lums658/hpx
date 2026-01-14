# Examples Gallery

Real-world application examples demonstrating HPXPy capabilities.

## Scientific Computing

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Monte Carlo Pi
:link: monte_carlo_pi
:link-type: doc

Estimate pi using parallel random sampling. Classic demonstration of embarrassingly parallel computation.

**Topics:** Random numbers, reduction, parallel speedup
:::

:::{grid-item-card} Monte Carlo Simulation
:link: monte_carlo
:link-type: doc

General Monte Carlo methods for numerical integration and probability estimation.

**Topics:** Random sampling, statistical analysis
:::

:::{grid-item-card} Heat Diffusion
:link: heat_diffusion
:link-type: doc

Solve the 2D heat equation using finite difference methods with parallel stencil operations.

**Topics:** PDEs, stencil operations, 2D arrays
:::

:::{grid-item-card} Parallel Integration
:link: parallel_integration
:link-type: doc

Numerical integration using parallel quadrature methods.

**Topics:** Numerical methods, reduction
:::

::::

## Finance

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Black-Scholes
:link: black_scholes
:link-type: doc

Option pricing using the Black-Scholes model with parallel evaluation across strike prices.

**Topics:** Financial modeling, vectorized math
:::

::::

## Machine Learning

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Neural Network
:link: neural_network
:link-type: doc

Simple feedforward neural network implemented with parallel matrix operations.

**Topics:** Matrix multiplication, activation functions
:::

:::{grid-item-card} K-Means Clustering
:link: kmeans_clustering
:link-type: doc

Parallel K-means clustering algorithm for unsupervised learning.

**Topics:** Clustering, iterative algorithms, distance computation
:::

:::{grid-item-card} ALS Recommender
:link: als_recommender
:link-type: doc

Alternating Least Squares for collaborative filtering recommendation systems.

**Topics:** Matrix factorization, iterative solvers
:::

:::{grid-item-card} LDA Topic Modeling
:link: lda_topic_modeling
:link-type: doc

Latent Dirichlet Allocation for discovering topics in text corpora.

**Topics:** NLP, probabilistic modeling
:::

::::

## Image Processing

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Image Processing
:link: image_processing
:link-type: doc

Parallel image filters and transformations including blur, sharpen, and edge detection.

**Topics:** 2D arrays, convolution, parallel transforms
:::

::::

## Distributed Computing

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Distributed Analytics
:link: distributed_analytics
:link-type: doc

Large-scale data analytics across multiple localities with collective operations.

**Topics:** Multi-locality, all_reduce, distributed arrays
:::

:::{grid-item-card} Distributed Reduction
:link: distributed_reduction
:link-type: doc

Distributed parallel reduction patterns across compute nodes.

**Topics:** Collectives, multi-node scaling
:::

:::{grid-item-card} Multi-Locality Demo
:link: multi_locality
:link-type: doc

Introduction to multi-locality programming with HPXPy launcher.

**Topics:** Locality management, distributed execution
:::

:::{grid-item-card} Distribution Demo
:link: distribution_demo
:link-type: doc

Demonstration of data distribution strategies across localities.

**Topics:** Data partitioning, load balancing
:::

::::

## Performance

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Scalability Analysis
:link: scalability
:link-type: doc

Measure and analyze parallel scaling efficiency with varying thread counts and problem sizes.

**Topics:** Benchmarking, strong/weak scaling
:::

::::

## Running Examples

### Prerequisites

```bash
pip install jupyterlab matplotlib numpy
```

### Run Locally

```bash
cd hpx/python/examples
jupyter lab
```

### Run from Documentation

Each example can be downloaded as a notebook using the download button at the top of the page.

```{toctree}
:maxdepth: 1
:hidden:

monte_carlo_pi
monte_carlo
heat_diffusion
parallel_integration
black_scholes
neural_network
kmeans_clustering
als_recommender
lda_topic_modeling
image_processing
distributed_analytics
distributed_reduction
multi_locality
distribution_demo
scalability
```
