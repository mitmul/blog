# Semi-supervised Classification with Graph Convolutional Networks

## Graph Laplacian

- A graph $\mathcal\{G} = \\{ \mathcal\{V}, \mathcal\{E} \}$ which has $N$ nodes.
- $A \in \mathbb\{R}^\{N \times N}$ is the adjacency matrix.
- $D \ (D_\{ii} = \sum_j A_\{ij})$ is the degree matrix.

The unnormalized graph Laplacian is defined as following.

$$
\Delta = D - A
$$

The normalized graph Laplacian is defined as following.

$$
L = I_N - D^\{-\frac\{1}\{2}} A D^\{-\frac\{1}\{2}}
$$

## Graph Convolutional Network

- $\tilde\{A} = A + I_N$ is the adjacency matrix of the undirected graph $\mathcal\{G}$ with self-connections, where $I_N$ is the identity matrix.
- $\tilde\{D}_\{ii} = \sum_j \tilde\{A}_\{ij}$ is the degree matrix calculated from $\tilde\{A}$.
- $\sigma(\cdot)$ denotes an activation function.
- $H^\{(l)}$ is the activation at $l$-th layer and $H^0$ means $X$.

Consider a multi-layer Graph Convolutional Network (GCN) with the following layer-wise propagation rule:

$$
H^\{(l+1)} = \sigma \left(
\tilde\{D}^\{-\frac\{1}\{2}}
\tilde\{A}
\tilde\{D}^\{-\frac\{1}\{2}}
H^\{(l)}
W^\{(l)}
\right)
$$

## Spectral Graph Convolutions

- $x \in \mathbb\{R}^N$ is a signal constructed from scalar values on every nodes.
- Let $g_\{\theta} = \{\rm diag}(\theta)$ be a filter parameterized by $\theta \in \mathbb\{R}^N$.
- We consider spectral convolutions on the graph $\mathcal\{G}$ is the multiplication of $x$ and $g_\theta$ **on Fourier domain**.
- 
