---
layout:  post
title:  "Neural Relational Inference for Interacting Systems"
mathjax: true
---

["Neural Relational Inference for Interacting Systems"](https://arxiv.org/abs/1802.04687)で使われていたGraph Neural Networksについての自分用メモ．

## Message Passing Neural Networksの計算方法

EncoderとDecoderそれぞれの中での計算に，受信ノードと送信ノードのIDを1-hotベクトルにして並べたものがよく用いられる．

まず初期状態では全ノードが自分自身への接続を除いて全て互いに繋がっているとすると，接続行列は非対角行列になる．これは，NumPyを用いて以下のように作れる．

```python
import numpy as np

num_nodes = 5

off_diagonal_mat = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
```

この行列はこうなる．

```python
array([[0., 1., 1., 1., 1.],
       [1., 0., 1., 1., 1.],
       [1., 1., 0., 1., 1.],
       [1., 1., 1., 0., 1.],
       [1., 1., 1., 1., 0.]])
```

次に，行を送信者（sender），列を受信者（receiver）と考え，この行列の$(i, j)$要素が1であるとは，ノード$i$からノード$j$に向かう接続がある，すなわち$i$が送信し$j$が受信しているという風に解釈してみる．すると，送信者の一覧と受信者の一覧は，

```python
sender, receiver = np.where(off_diagonal_mat)
```

で取り出せて，sender, receiverの中身は順に，以下のようになる．

```python
array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3])
```

これは最初の送信者はノード0，受信者はノード1，つまり非対角行列の$(0, 1)$が1であったことを意味し，以下同様，という感じ．

これを1-hotベクトルにする．

```python
rel_send = np.eye(len(np.unique(sender)))[sender]
rel_rec = np.eye(len(np.unique(receiver)))[receiver]
```

中身は以下のようになっている．長いので`rel_send`の一部だけ表示．

```python
array([[1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       ...
```

この1-hotベクトルにした送信者ノードIDの一覧と受信者ノードIDの一覧（それぞれ`rel_send`, `rel_rec`）を色々なところで使う．

## Encoder

### (1) 埋め込みベクトルを作る
$$
\{\bf h}_j^\{(1)} = f_\{\rm emb}(\{\bf x}_j) \label\{eq:encode_embedding}
$$

$\{\bf x}_j \in \mathcal\{R}^\{d_\{\rm init}}$は、$j$番目のノード特徴量の初期値．この論文では位置と速度をconcatしたもの．$f_\{\rm emb}$は$\mathcal\{R}^\{d_\{\rm init}} \rightarrow \mathcal\{R}^\{d_\{\rm emb}}$をやるMLPかCNN．$\{\bf h}_j^\{(1)} \in \mathcal\{R}^\{d_\{\rm emb}}$を出力する．

まず最初の入力はミニバッチ次元も含めると`[batch_size, num_nodes, timesteps, feature_dims]`という形式になっている．

### (2) ノード特徴からエッジ特徴を作る
$$
v \rightarrow e: \hspace\{2em}
\{\bf h}_\{(i, j)}^\{(1)} = f_e^\{(1)}([\{\bf h}_i^\{(1)}, \{\bf h}_j^\{(1)}]) \label\{eq:encode_node2edge}
$$

### (3) エッジ特徴からノード特徴を更新
$$
e \rightarrow v: \hspace\{2em}
\{\bf h}_j^\{(2)} = f_v^\{(1)}(\sum_\{i \neq j} \{\bf h}_\{(i, j)}^\{(1)}) \label\{eq:encode_edge2node}
$$

### (4) ノード特徴からエッジ特徴を更新
$$
v \rightarrow e: \hspace\{2em}
\{\bf h}_\{(i, j)}^\{(2)} = f_e^\{(2)}([\{\bf h}_i^\{(2)}, \{\bf h}_j^\{(2)}]) \label\{eq:encode_node2edge2}
$$

## Decoder

### (1) 予測エッジをtimestepsだけrepeatする
入力はノード特徴（データセット由来）：`[batch_size, num_nodes, timesteps, feature_dim]`と，予測エッジ：`[batch_size, num_edges, edge_types]`だが，**NRIでは時系列方向にエッジの意味は変わらないと仮定する**ので，予測エッジのテンソルに2nd orderを新たに挿入して，`timesteps`次元になるようにbroadcastする．

```python
import chainer.functions as F

rel_type = F.broadcast_to(
    rel_type[:, None, :, :],
    [batchsize, timesteps, num_edges, edge_types]
)
```

### (2) 予測ステップ数ごとに区切る
```python
# Only take n-th timesteps as starting points (n: pred_steps)
last_pred = inputs[:, 0::pred_steps, :, :]
curr_rel_type = rel_type[:, 0::pred_steps, :, :]
# NOTE: Assumes rel_type is constant (i.e. same across all time steps).
```

入力ノード特徴の時間方向の長さ`timesteps`は学習時に予測させるステップ数`pred_steps`よりも長くなるようにしておき，時間軸にそって`pred_steps`おきにノード特徴を取り出して`pred_steps`だけ予測させることで，一つのノード特徴列を複数のシーケンスに分割して各シーケンスごとに最初の位置・速度（ノード特徴）だけ与えてあとを予測させるという形にする．例えば，`timesteps`が0~49（50ステップ）のノード特徴列がデータセットから入ってきたとして，それを`pred_steps`（著者実装では10）ごとに切って，0:9, 10:19, 20:29, 30:39, 40:49の５つのシーケンスにし，それぞれのシーケンスごとに最初のノード特徴だけ与えて，あとを予測させるという風にする．（ちょっとTeacher Forcingっぽい？）たぶん49ステップとか一気に予測させると誤差が大きすぎて厳しいので，10ステップ程度にしておきたいが入力データを有効活用したいからこんな感じの実装になっているのでは．

### (3) 1ステップ先を予測
まず各シーケンスのあたまのノード特徴をエッジの受信者側，送信者側のノードについて取り出して，concatする．

```python
receivers = F.matmul(rel_rec, single_timestep_inputs)
senders = F.matmul(rel_send, single_timestep_inputs)
pre_msg = F.concat([receivers, senders], axis=-1)
```

ここで，`single_timestep_inputs`は`[batch_size, num_sequences, num_nodes, feature_dims]`で，各シーケンスのあるタイムステップのノード特徴を羅列したもの．それぞれ，`rel_rec`や`rel_send`とmatmulすることで受信者ノード・送信者ノードの特徴を取り出せる．concatは`feature_dims`にそってされるので，`pre_msg`のshapeは`[batch_size, num_sequences, num_edges, 2 * feature_dims]`になる．

`single_timestep_rel_type`は`[batch_size, num_sequences, num_edges, edge_types]`で各シーケンスのあるタイムステップにおける全エッジのエッジタイプを羅列したもの．

