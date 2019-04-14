# Probability-based Detection Quality

- $f$ 番目のフレームにおける $i$ 個目の ground truth $\mathcal\{G}_i^f$
- $f$ 番目のフレームにおける $j$ 個目の detection $\mathcal\{D}_j^f$

この 2 つのペアに対する pairwise probability-based detection quality (pPDQ) はその幾何平均で定義される．

$$
\{\rm pPDQ}(\mathcal\{G}_i^f, \mathcal\{D}_j^f)
= \sqrt\{Q_S(\mathcal\{G}_i^f, \mathcal\{D}_j^f) \cdot Q_L(\mathcal\{G}_i^f, \mathcal\{D}_j^f)}
$$

ここで $Q_S$ は spatial quality．$Q_L$ は label quality．

## Spatial quality 

spatial quality は foreground loss $L_\{FG}$ と background loss $L_\{BG}$ を使って以下のように定義される．

$$
Q_S(\mathcal\{G}_i^f, \mathcal\{D}_j^f)
= \exp \left( - (L_\{FG}(\mathcal\{G}_i^f, \mathcal\{D}_j^f) + L_\{BG}(\mathcal\{G}_i^f, \mathcal\{D}_j^f) \right)
$$

## Label quality

label quality はいかに効率的に detection がその物体が何であるかを同定しているかを測る．

$$
Q_L(\mathcal\{G}_i^f, \mathcal\{D}_j^f) = \{\bf l}_j^f (\hat\{c}_i^f)
$$

## detection-object ペアの最適割当

- すべての detection-object ペアに対して pPDQ を計算して，その表に対して Hungarian algorithm を使って最適割当を決める．
- すると合計の pPDQ スコアが最大になるような組み合わせが出てくる．
- Hungarian algorithm の結果求まった割当における pPDQ が 0 でないものの数を $N_\{\rm TP}^f$ とする．
- 
