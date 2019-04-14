# Linear Algebra Notes

## Invertible matrix

- If there exists a square matrix $B$ of order $n$ s.t. $AB = I = BA$, the matrix $A$ is an invertible matrix (= nonsingular or nondegenerate matrix).

## Eigenvalues / Eigenvectors

- A linear transformation is just scaling along with the axes in the original space if the real matrix which performs that transformation is diagonal.
- Eigenvectors can be thought as axes of such directions where the transformation can be seen as a scaling.
- How much scale the matrix does along with the eigenvectors are eigenvalues.
- For a $n$-by-$n$ matrix $A$, there exists $\lambda, {\bf p}$ s.t. $A {\bf p} = \lambda {\bf p}$ which are eigenvalues and eigenvectors, respectively. This is the definition of eigenvalues and eigenvectors.
- Let's think about a linear system: ${\bf x}(t) = A {\bf x}(t - 1)$. We need to know the eigenvalues of this transition matrix $A$ because if it's positive semidefinite (all eigenvalues are $\geq 0$), it means ${\bf x}(t)$ will diverge.
- Now we transform ${\bf x}(t)$ with another matrix $P$ which is invertible: ${\bf x}(t) = P {\bf y}(t)$. This can be deformed into the following.
  $$
  \begin{align}
  {\bf x}(t) &= P {\bf y}(t) \\
  \Leftrightarrow {\bf y}(t) &= P^{-1} A {\bf x}(t - 1) \\
  \Leftrightarrow {\bf y}(t) &= P^{-1} A P {\bf y}(t - 1)
  \end{align}
  $$
  Here, let's put $\Lambda \equiv P^{-1} A P$.
- If we can make $\Lambda$ diagonal,
  $$
  \begin{align}
  {\bf y}(t) = \Lambda^t {\bf y}(0)
  \end{align}
  $$
  holds, and it can be deformed into
  $$
  \begin{align}
  {\bf y}(t) &= \Lambda^t {\bf y}(0) \\
  \Leftrightarrow P^{-1} {\bf x}(t) &= \Lambda^t P^{-1} {\bf x}(0) \\
  \Leftrightarrow {\bf x}(t) &= P \Lambda^t P^{-1} {\bf x}(0)
  \end{align}
  $$
- Now we assume that $\Lambda$ is a diagonal matrix, and its diagonal elements are the eigenvalues of $A$. Because,
  $$
  \begin{align}
  \Lambda &= P^{-1} A P \\
  \Leftrightarrow P \Lambda &= A P \\
  \Leftrightarrow (\lambda_1 {\bf p}_1, \dots, \lambda_n {\bf p}_n)
  &= (A {\bf p}_1, \dots, A {\bf p}_n)
  \end{align}
  $$
  which represents
  $$
  \begin{align}
  \lambda_1 {\bf p}_1 &= A {\bf p}_1 \\
  &\vdots \\
  \lambda_n {\bf p}_n &= A {\bf p}_n
  \end{align}
  $$
  and these are exactly the definition of eigenvalues and eigenvectors of the matrix $A$.
- One way to calculate the actual values of eigenvalues and eigenvectors is to solve this **characteristic polynomial** of $A$:
  $$
  \phi_A(\lambda) \equiv \det \left( \lambda I - A \right) = 0
  $$
