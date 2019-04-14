# Graph R-CNN for Scene Graph Generation

画像からシーングラフを生成するタスクを考える．シーン中のすべての物体間のあらゆる可能な関係を考慮するのは現実的でないから，既存手法の中にはランダムに物体間の間に張られるエッジをサンプルするものがある．

$$
P(S | {\bf I}) =
P({\bf V} | {\bf I})
P({\bf E} | {\bf V}, {\bf I})
P({\bf R}, {\bf O} | {\bf V}, {\bf E}, {\bf I})
$$

- ${\bf I}$: 入力画像
- ${\bf V}$: 入力画像中の各物体領域に対応するノードの集合
- ${\bf E}$: 物体間の関係（またはエッジ）． 接続のありなし
- ${\bf O}$: 物体のクラスラベル
- ${\bf R}$: 物体間の関係のラベル

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
- $n (n - 1)$ 個全ての directional なペア $\{ {\bf p}^o_i, {\bf p}^o_i | i \neq j\}$ をスコア付けして，relatedness を $s_{ij} = f({\bf p}^o_i, {\bf p}^o_j$ で計算
- $f$ は relatedness 関数．${\bf p}^o_i$ と ${\bf p}^o_j$ を concat したものを入力に取る MLP とかを使う単純な方法も考えられる
- ただ，それを全部のペアに実際やると計算が大変すぎるので，非対称なカーネル関数を使う：
  $$
  f({\bf p}^o_i, {\bf p}^o_j) = 
  \langle \Phi({\bf p}^o_i)), \Psi({\bf p}^o_j) \rangle, i \neq j
  $$
- つまり directional なエッジの根本と先端で別々のカーネル関数を適用して，その結果の内積を取ることで relatedness にする
- カーネル関数にはそれぞれ MLP を使う
- relatedness 推定したら降順にソートしてトップ $K$ ペアを残す
- それらに NMS を適用して重複したペアを取り除く
- 各ペアは bounding box を持っている
- 各ペアの順番 (エッジの方向) には意味がある
- ペアとペアの間の overlap は以下のように計算する．ペア $\{u, v\}$ と $\{p, q\}$ があったとき
  $$
  {\rm IoU}(\{u, v\}, \{p, q\}) =
  \frac
  {I(r^o_u, r^o_p) + I(r^o_v, r^o_q)}
  {U(r^o_u, r^o_p) + U(r^o_v, r^o_q)}
  $$
  ただし $I$ は intersection，$U$ は union を表す
- この Pairwise NMS を行ったあとに残った $m$ 個のペアは意味のある関係の候補の集合 ${\bf E}$ とおく
- これで一番最初よりだいぶ選抜された $\mathcal{G} = ({\bf V}, {\bf E})$ が得られた
- これら $m$ 個のペアに対して，始点・終点のノードの region の union に対して visual feature をとりだしたものを $X^r = \{{\bf x}^r_1, \dots, {\bf x}^r_m\}$ とおく

### 結局得られたもの

1. 選抜された ${\bf V}, {\bf E}$
2. 各ペアの union に対する visual feature ${\bf X}^r$

## Attentional Graph Convolutional Network

### まずは Vanilla GCN

- 各ノード $i$ が特徴ベクトル ${\bf z}_i \in \mathbb{R}^d$ を持っている
- ノード $i$ と接続がある近隣のノードの集合 $\{ {\bf z}_j | j \in \mathcal{N}(i) \}$ を学習で決定する重み行列 ${\bf W}$ で線形変換する
- 変換した隣接ノードの特徴ベクトルに事前に決定されている $\boldsymbol\alpha$ を掛ける
- その結果に非線形変換を施す
- まとめると
  $$
  {\bf z}_i^{(l+1)} = \sigma \left(
  {\bf z}_i^{(l)} +
  \sum_{j \in \mathbb{N}(i)}
  \alpha_{ij} {\bf W} {\bf z}_j^{(l)}
  \right)
  $$


まず ${\bf I}$ から ${\bf V}$ を推定．これは物体らしい