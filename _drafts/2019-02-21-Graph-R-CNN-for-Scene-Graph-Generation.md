---
layout:  post
title:  "Graph R-CNN for Scene Graph Generation"
mathjax: true
---

# Graph R-CNN for Scene Graph Generation

画像からシーングラフを生成するタスクを考える．シーン中のすべての物体間のあらゆる可能な関係を考慮するのは現実的でないから，関係性のあるなしについての度合いを学習できる Attentional GCN を使って，候補となる relation を間引く．

## アプローチ

- 画像からシーングラフを出力するモデルを以下のように分解
  $$
  \begin{equation}
  P(S | {\bf I}) =
  P({\bf V} | {\bf I})
  P({\bf E} | {\bf V}, {\bf I})
  P({\bf R}, {\bf O} | {\bf V}, {\bf E}, {\bf I})
  \end{equation}
  $$
    - $S$: シーングラフ．$({\bf V}, {\bf E}, {\bf O}, {\bf R})$ のこと
    - ${\bf I}$: 入力画像
    - ${\bf V}$: 入力画像中の各物体領域に対応するノードの集合
    - ${\bf E}$: 物体間の関係（またはエッジ）． 接続のありなし
    - ${\bf O}$: 物体のクラスラベル
    - ${\bf R}$: 物体間の関係のラベル
- $P({\bf V} | {\bf I})$: 物体領域候補の抽出
- $P({\bf E} | {\bf V}, {\bf I})$: 関係がありそうなノード間のエッジの抽出
- $P({\bf R}, {\bf O} | {\bf V}, {\bf E}, {\bf I})$: グラフラベリング（エッジに関係の種類をラベル付け）

## Object proposal

- まず Faster R-CNN で $n$ 個の object proposal を作成
- 各 object proposal $i$ は
    - 領域: $r^o_i [x_i, y_i, w_i, h_i]$
    - プーリング後の特徴ベクトル: $x^o_i$ ($d$ 次元)
    - 推定されたクラスラベル確率: $p^o_i$ ($k$ = クラス数次元)
- クラス $C = \{1, \dots, k\}$
- これらを $n$ 個並べた行列
    - $R^o \in \mathbb{R}^{n \times 4}$
    - $X^o \in \mathbb{R}^{n \times d}$
    - $P^o \in \mathbb{R}^{n \times |C|}$

## Relation Proposal Network

- $n$ 個の object proposal があると，それらの間の接続は $O(n^2)$ 個考えられるが，ほとんどの物体間には関係がない
- 関係がなさそうな物体間のエッジを間引くために，まず relation proposal network (RePN) で物体間の relatedness を推定する
- 推定されたクラスラベルの分布 $P^o$ を使って relatedness を推論する
- 「クラス - 関係」という prior を学習する
    - どの物体かによってどの物体との間にどういう関係を持ちやすいかの偏りがあるはず．それを使う
- $n (n - 1)$ 個全ての directional なペア $ \{ {\bf p}^o_i, {\bf p}^o_i | i \neq j \} $ をスコア付けして，relatedness を $s_{ij} = f({\bf p}^o_i, {\bf p}^o_j)$ で計算
- $f$ は relatedness 関数．${\bf p}^o_i$ と ${\bf p}^o_j$ を concat したものを入力に取る MLP とかを使う単純な方法も考えられる
- ただ，それを全部のペアに実際やると計算が大変すぎるので，非対称なカーネル関数を使う：
  $$
  \begin{equation}
  f({\bf p}^o_i, {\bf p}^o_j) =
  \langle \Phi({\bf p}^o_i), \Psi({\bf p}^o_j) \rangle, i \neq j
  \end{equation}
  $$
- つまり directional なエッジの根本と先端で別々のカーネル関数を適用して，その結果の内積を取ることで relatedness にする
- カーネル関数にはそれぞれ MLP を使う
- relatedness 推定したら降順にソートしてトップ $K$ ペアを残す
- それらに NMS を適用して重複したペアを取り除く
- 各ペアは bounding box を持っている
- 各ペアの順番 (エッジの方向) には意味がある
- ペアとペアの間の overlap は以下のように計算する．ペア $\{u, v\}$ と $\{p, q\}$ があったとき
  $$
  \begin{equation}
  {\rm IoU}(\{u, v\}, \{p, q\}) =
  \frac
  {I(r^o_u, r^o_p) + I(r^o_v, r^o_q)}
  {U(r^o_u, r^o_p) + U(r^o_v, r^o_q)}
  \end{equation}
  $$
  ただし $I$ は intersection，$U$ は union を表す
- この Pairwise NMS を行ったあとに残った $m$ 個のペアは意味のある関係の候補の集合 ${\bf E}$ とおく
- これで一番最初よりだいぶ選抜された $\mathcal{G} = ({\bf V}, {\bf E})$ が得られた
- これら $m$ 個のペアに対して，始点・終点のノードの region の union に対して visual feature をとりだしたものを $ X^r = \{ {\bf x}^r_1, \dots, {\bf x}^r_m \} $ とおく

### 実装方法

1. class_probs (batch_size, num_rois, num_classes) と region_proposals (batch_size, num_rois, 4) を受け取る
2. 全 RoI が密結合しているとして subject と object (sender と receiver) のノード ID (1-hot vector) が並んだ配列を作る（rel_subj, rel_obj: (num_edges, num_rois) という形の配列．2 次元目が num_rois なのは 1-hot vector だから）
3. 

### 結局得られるもの

1. 選抜された ${\bf V}, {\bf E}$
2. 各ペアの union に対する visual feature ${\bf X}^r$

## Attentional Graph Convolutional Network

### Vanilla GCN

- まず attentional じゃない vanilla GCN について
- 各ノード $i$ が特徴ベクトル ${\bf z}_i \in \mathbb{R}^d$ を持っている
- ノード $i$ と接続があるノード（隣接ノード）の集合 $ \{ {\bf z}_j | j \in \mathcal{N}(i) \} $ を学習で決定する重み行列 ${\bf W}$ で線形変換する
- 変換した隣接ノードの特徴ベクトルに事前に決定されている $\boldsymbol\alpha$ を掛ける
- その結果に非線形変換を施す
- 以上の操作をまとめると
  $$
  \begin{equation}
  {\bf z}_i^{(l+1)} = \sigma \left(
  {\bf z}_i^{(l)} +
  \sum_{j \in \mathcal{N}(i)}
  \alpha_{ij} {\bf W} {\bf z}_j^{(l)}
  \right)
  \end{equation}
  $$
- 上式は，${\bf W} \in \mathbb{R}^{d \times d}$ に ${\bf z}_j^{(l)}$ を横に並べた ${\bf Z}^{(l)} \in \mathbb{R}^{d \times | \mathcal{N}(i) |}$ を右から掛けて変換したものに $\boldsymbol\alpha_i = [\alpha_{i1}, \dots, \alpha_{ij}]^{\rm T} \in \mathbb{R}^{| \mathcal{N}(i) |}$ を掛ける，と書き直すと
  $$
  \begin{equation}
  {\bf z}_i^{(l+1)} =
  \sigma \left(
  {\bf z}_i^{(l)} +
  {\bf W} {\bf Z}^{(l)} \boldsymbol\alpha_i
  \right)
  \end{equation}
  $$
  と書ける．ただし，論文中では違う式変形がされている
- 論文中では，${\bf Z}$ を全ノードの特徴ベクトルを並べたもの（${\bf Z} \in \mathbb{R}^{d \times n}$）とし，また $\boldsymbol\alpha_i$ は $[0, 1]$ の範囲の値の $n$ 次元ベクトル（$\boldsymbol\alpha_i \in [0, 1]^{n}$）としており，かつ $\boldsymbol\alpha_i$ の要素にはノード $i$ と接続を持たないところで $0$，また $\alpha_{ii} = 1$ という制約を与えている
- そして論文中の式(5)は以下．
  $$
  {\bf z}_i^{(l+1)} = \sigma \left(
  {\bf W} {\bf Z}^{(l)} \boldsymbol\alpha_i
  \right)
  \tag{5'}
  $$
- こうすると ${\bf W}{\bf Z}^{(l)}$ が全ノードの特徴ベクトルを線形変換したものになるので，${\bf z}_i^{(l)}$ にも線形変換がかかっていることになるのに注意．この点において式(4)と equivalent ではない
- なので，$\boldsymbol\alpha_i$ は，$\alpha_{ii} = 1$ とするのではなく $\alpha_{ii} = 0$ として，${\bf z}_i^{(l)}$ を加える項を足すと，式(4)と等しくなる
  $$
  {\bf z}_i^{(l+1)} = \sigma \left(
  {\bf z}_i^{(l)} + {\bf W}{\bf Z}^{(l)} \boldsymbol\alpha_i
  \right)
  \tag{5''}
  $$
  （ここでの $\boldsymbol\alpha_i$ は式(5')のものとは異なることに注意；$\alpha_{ii} = 0$ になっている）
- しかし，どうやらこの論文の式(4)の方が間違っていて，式(5)が正しいもよう
- 論文中には [13] として Kipf. et al. の "Semi-supervised classification with Graph Convolutional Network" が引用されているが，式(4)は恐らく（引用されていない）別論文 "Modeling relational data with Graph Convolutional Network" (Schlichtkrull et al.) 中の式(2):
  $$
  h_i^{(l+1)} = \sigma \left(
  \sum_{r \in \mathcal{R}}
  \sum_{j \in \mathcal{N}_i^r}
  \frac{1}{c_{i,r}}
  W_r^{(l)} h_j^{(l)} + W_0 h_i^{(l)}
  \right)
  $$
  に由来している．そう考えると，式(4)は正しくは
  $$
  {\bf z}_i^{(l+1)} = \sigma \left(
  {\bf W}_0 {\bf z}_i^{(l)} +
  \sum_{j \in \mathcal{N}(i)} \alpha_{ij} {\bf W}_r {\bf z}_j^{(l)}
  \right)
  $$
  であると思われる．そうするとこれは上の式(5')のように書けるので，結局本論文中の式(5)（上の式(5')）を使うのが正しいと思われる．

### Attentional GCN (aGCN)

- Vanilla GCN では $\boldsymbol\alpha_i$ は固定だが，学習したいので拡張する
- $\boldsymbol\alpha_i$ を 2 つのノード特徴を concat したものを入力にとる 2 層 MLP を使って予測する
  $$
  \begin{align}
  u_{ij} &= {\bf w}_h^{\rm T} \sigma \left(
  {\bf W}_a [ {\bf z}_i^{(l)}, {\bf z}_j^{(l)} ]
  \right) \\
  \boldsymbol\alpha_i &= {\rm softmax}({\bf u}_i)
  \end{align}
  $$
  ここで $[ \cdot, \cdot ]$ は concat の意味．また，${\bf W}_a \in \mathbb{R}^{h \times 2d}$，${\bf w}_h \in \mathbb{R}^h$，${\bf u}_i \in \mathbb{R}^{| \mathcal{N}(i) |}$．
- このようにして 2 つのノードペアに対して学習可能な重みを使って計算される

### aGCN を Scene Graph に使う

- ここで，物体（RoI）がノード，物体間の関係がエッジ，というこれまでの設定から，関係もノードとして扱う形に拡張する
- すると扱う接続関係は
    - subject <-> subject
    - subject <-> object
    - object <-> object
    - subject <-> relationship
    - object <-> relationship
    - relationship <-> relationship
- また，これらは directional に扱われる．ある relationship に対する subject のもつ情報の重要性は relationship が subject が何かに対して持つものと同じではないため
- 考えうる接続関係それぞれについて，別々の線形変換を学習する
- それぞれの線形変換は，ノードタイプ $a$ から $b$ への変換に用いられるものを ${\bf W}^{ab}$ と書くことにする．また書くノードタイプには以下の文字を用いる
    - $s$: subjects
    - $o$: objects
    - $r$: relationships
- 物体ノードの特徴ベクトルを並べたものを ${\bf Z}^o$，関係ノードの特徴ベクトルを並べたものを ${\bf Z}^r$ とする
- そして，
- このとき，ある物体ノードの特徴ベクトル ${\bf z}_i^o$ は以下のように更新される
  $$
  \begin{equation}
  {\bf z}_i^o =
  \sigma \left(
  {\bf W}^{oo} {\bf Z}^o \boldsymbol\alpha^{oo} +
  {\bf W}^{sr} {\bf Z}^{r} \boldsymbol\alpha^{sr} +
  {\bf W}^{or} {\bf Z}^r \boldsymbol\alpha^{or}
  \right)
  \end{equation}
  $$
  - $\boldsymbol\alpha_{ii}^{oo} = 1$ とする
  - ${\bf W}^{oo} {\bf Z}^o \boldsymbol\alpha^{oo}$ : 別の物体からのメッセージ
  - ${\bf W}^{sr} {\bf Z}^{r} \boldsymbol\alpha^{sr} + {\bf W}^{or} {\bf Z}^r \boldsymbol\alpha^{or}$ : 隣接している関係ノードからのメッセージ
- 