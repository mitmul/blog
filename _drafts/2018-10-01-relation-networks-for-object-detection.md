---
layout:  post
title:  "Relation Networks for Object Detection"
mathjax: true
---

## Object Relation Module

### Scaled Dot-Product Attention

まず["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)で提案されたScaled Dot-Product Attention (SDPA)について理解する必要がある。

1. それぞれ$d_k$次元であるqueryとkey、および$d_v$次元であるvalueを考える。
2. まずqueryとすべてのkeyの内積を計算し、query-key間のsimilarityを算出する。
3. そのあと$1/\sqrt{d_k}$によるスケーリングを行ってからsoftmaxを適用する。
   1. このスケーリングには、次元数が大きくなると内積の値が巨大になるため、softmax関数の勾配が非常に小さくなる（logistic sigmoidの勾配がサチった領域を思い浮かべる）というのを防ぎたいという意図がある。
   2. また、queryとkeyの内積によって類似度が測られたあと、softmaxを適用することには、queryに似たkeyの列番号において大きな値を持つようなベクトルを出力することになるので、類似度の高いキーのインデックスをソフトに表すようなものになる。
4. ソフトなキーインデックスベクトルとvalue行列 $V$ を掛け合わせて、attentionを得る。

出力の計算は以下の式で表される。queryを${\bf q}$、すべてのkeyを並べた行列を$K$、valueを並べたものを行列$V$とすると、

$$
v^{\rm out} = {\rm softmax}\left( \frac{ {\bf q} K^t}{\sqrt{d_k}} \right) V.
$$

### Object Relation

ではObject relationはどうやって計算するか。まず、objectはgeometric feature ${\bf f}_G$とappearance feature ${\bf f}_A$によって構成されていると考えてみる。今回は、${\bf f}_G$はシンプルに４次元のbounding boxで、${\bf f}_A$はタスクによって変えることにする。

$N$個の物体が与えられたとして：$\left\{ ( {\bf f}^n_A, {\bf f}^n_G) \right\}^N_{n=1} $ 、$n$個目の物体とすべての物体との間のrelation feature ${\bf f}_R(n)$は、以下のように計算される：

$$
{\bf f}_R(n) = \sum_m w(m, n) \cdot (W_V \cdot {\bf f}^m_A)
$$

* 論文中では$w^{mn}$と書かれているが、オーラル発表資料では$w(m, n)$と書かれておりこちらの方がわかりやすいので後者を採用した。

relation featureは他のすべての物体のappearance featureを$W_V$で線形変換したもの（$W_V$はScaled Dot-Product Attentionにおける$V$に相当する）の重み付け和になっていて、重みに相当する$w(m, n)$は、物体$n$が物体$m$から受ける影響の強さを表しており、以下のように定義される：

$$
w(m, n) = \frac{w_G(m, n) \cdot \exp (w_A(m, n))}{\sum_k w_G(k, n) \cdot \exp(w_A(k, n))}
$$

appearance weight $w_A(m, n)$は以下のような内積で計算される：

$$
w_A(m, n) = \frac{ {\rm dot}(W_K {\bf f}^m_A, W_Q {\bf f}^n_A)}{\sqrt{d_k}}
$$

ここで$W_K$と$W_Q$は、SDPAにおける$K$と$Q$と似たような役割を果たす。これらは、${\bf f}^m_A$と${\bf f}^m_A$というfeatureをそれぞれどのくらいマッチしているかを図るためのsubspaceにprojectする。projectしたあとの特徴量の次元は$d_k$である。

geometry weight $w_G(m, n)$は、以下のように計算される：

$$
w_G(m, n) = \max \left\{ 0, W_G \cdot \varepsilon_G ({\bf f}^m_G, {\bf f}^n_G) \right\}
$$

２ステップある。まず、２つの物体のgeometry featuresは高次元表現に埋め込まれる（$\varepsilon_G$がやる）。この$\varepsilon_G$について細かく説明する。まずtranslationとscale変換にinvariantにするために、４次元のrelative geometry featureを計算する。これは以下のようになる：

$$
\left(
\log\left( \frac{|x_m - x_n|}{w_m} \right),
\log\left( \frac{|y_m - y_n|}{h_m} \right),
\log\left( \frac{w_n}{w_m} \right),
\log\left( \frac{h_n}{h_m} \right)
\right)^T
$$

これをサイン・コサイン関数を使って高次元に埋め込む。次元ごとに異なる波長を使う。高次元に埋め込み後の次元数は$d_g$になる。





- GossipNetというのがduplicate removalをでかいネットワークでやっていたけどNMSとコンパラなのに計算コストがでかすぎた。realtion moduleはシンプルで一般的かつSoftNMSより性能が良い。
- attentionを物体検出に導入したいから、新しく物体間の位置関係を捉えるためのgeometric weightってのを提案する。これは並進移動普遍性がある。



```python
f_G = (xmin, ymin, xmax, ymax)
f_R[n] =
```


