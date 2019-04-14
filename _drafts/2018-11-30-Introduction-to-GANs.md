---
layout:  post
title:  "Introduction to GANs"
mathjax: true
---

Generative adversarial networks（GAN）は，一様分布や正規分布からランダムサンプルされたノイズを対象のデータ形式に変換する形で生成を担うGenerator（$G$）と，この$G$が生成したサンプルと学習データセットに含まれる実例を識別するように訓練されるDiscriminator（$D$）を，互いに競わせるようにして学習させることで，生成モデルを推定する方法である．本稿では，このGANの基礎およびGANを用いて画像生成を行ういくつかの手法を解説する．

## はじめに

対象を理解するために、それを生成する方法を模索することは，しばしば重要な知見をもたらしてくれる．“What I cannot create, I do not understand.”[^feynman] というリチャード・ファインマンの言葉があるが，作ることと理解することの間に重要な関係があることは我々も日常的に感ずることがあるだろう．深層学習を用いた画像認識技術の発展が著しいが，本稿では画像を認識する方法ではなく，画像を生成する方法に関して解説を行う．特に2014年，機械学習分野の国際会議であるNeural Information Processing Systems（NIPS）にてIan Goodfellowにより発表されて以来，様々な分野から大きな注目を集めているGenerative adversarial networks（GAN）[^GAN]をベースとした画像生成の手法について解説を試みる．まず生成モデルを研究する意義について触れ，GANの仕組みについて簡単に解説を行ってから，画像生成に用いられるGANの発展的手法について紹介し，最後に実装についても触れる．本稿では，紙面の関係上，機械学習に用いられる確率・統計の基礎や深層学習に用いられるニューラルネットワークについての基礎的な知識の解説は省略する．

## 生成モデルとGAN

自然界のデータには，その内容によって起こりやすさに偏りがある．例えば，人間がパスポートや免許証の申請に用いるような証明写真のデジタルデータを大量に集めてきたとすると，それらの画像の中央付近では，肌色に近い色のピクセルが存在する確率が，緑や青といった色のピクセルが存在する確率よりも明らかに高くなっているだろう．つまり，それらの画像から中央付近を適当に切り出してそこに存在するRGBの3次元ベクトルの値ごとの出現しやすさを返す関数を考えてみると，肌色：$(R, G, B) = (252, 226, 196)$付近で大きな値を返すような関数になることが予想される．もし，あるデータに関して，どのような値がどのような確率で現れるのかを表す確率分布が分かっているとするなら，これを用いて実際のデータと同じ分布に従う値をサンプルすることができる場合がある．このように，何らかの方法でデータが生成される仕組みを表現したものを生成モデルという．確率分布そのものが陽に求められなくても，観測された値からデータが従う確率分布を推定し，その推定に基づいてサンプルを抽出することができるようなモデルも，生成モデルの一種である．

生成モデルを学習する方法の一つには，データの確率分布を何らかのパラメータを持つモデル（例えば，平均と分散というパラメータを持つガウス関数など）で表現しておき，観測されたデータの分布$p_\{\rm data}$とこのモデルが表す分布$p_\{\rm model}$の間の差異が小さくなるように，モデルのパラメータを決定するといったものがある．例えば，その差異をKullback-Leibler （KL）ダイバージェンスによって測り，これを最小化する場合，それは観測データの尤度を最大化するようにパラメータを決定することと等しいので，最尤推定と呼ばれる．ここで，$p_\{\rm data}$の$p_\{\rm model}$に対するKLダイバージェンスは

$$
KL(p_\{\rm data} || p_\{\rm model}) = \int_\{-\infty}^\{\infty} p_\{\rm data}(x) \log \frac\{p_\{\rm data}(x)}\{p_\{\rm model}(x)} \{\rm d}x
$$

と定義される．これは，以下のように変形できる．

$$
\begin\{align}
KL(p_\{\rm data} \parallel p_\{\rm model})
&= \int_\{-\infty}^\{\infty} p_\{\rm data}(x) \log \frac\{p_\{\rm data}(x)}\{p_\{\rm model}(x)} \{\rm d}x \nonumber \\
&= \int_\{-\infty}^\{\infty} p_\{\rm data}(x) \log p_\{\rm data}(x) \{\rm d}x - \int_\{-\infty}^\{\infty} p_\{\rm data}(x) \log p_\{\rm model}(x) \{\rm d}x \nonumber
\end\{align}
$$

すると，この第1項は$p_\{\rm data}(x)$というデータ分布が変わらないとすれば定数であり，一方第2項は符号を除けば$\log p_\{\rm model}(x)$という値の期待値

$$
\mathbb\{E}_\{p_\{\rm data}} \left[ \log p_\{\rm model}(x) \right]
$$

であることがわかる．この$p_\{\rm data}$全体に渡る期待値計算を，実際に観測された$N$個のデータ$\\\{x_1, x_2, \dots, x_N\\}$を用いた$\log p_\{\rm model}(x)$の値の平均値で近似することを考える．すなわち実際に観測したサンプル＝データ集合全体の部分集合を用いて近似したデータ分布（これを経験分布という）を使った期待値計算で近似すると，

$$
\begin\{align}
\mathbb\{E}_\{p_\{\rm data}} \left[ \log p_\{\rm model}(x) \right] 
&\approx \frac\{1}\{N} \sum_\{i=1}^N \log p_\{\rm model}(x_i) \nonumber \\
&= \frac\{1}\{N} \log \prod_\{i=1}^N p_\{\rm model}(x_i) \nonumber
\end\{align}
$$

となる．このとき，$\prod_\{i=1}^N p_\{\rm model}(x_i)$は，観測されたデータ列$\\\{x_1, x_2, \dots, x_N\\}$の$p_\{\rm model}$を用いて測られる尤度（likelihood）を表しているから，上の式はこの尤度の対数をとったもの（対数尤度）であることがわかる．

結果，KLダイバージェンスはモデルに関係する部分だけを見ればこの対数尤度の符号を反転させたもの（負の対数尤度）となるから，KLダイバージェンスを最小にするということは，尤度を最大化するということに他ならないわけである．よって，最尤推定と呼ばれる．しかし，この最尤推定を用いて高次元データの生成モデルを学習するのは，非常に困難であった．次に，その理由を説明する．

冒頭の例では，画像の各ピクセルをRGBを意味する3次元空間上の1点であるとして考えたが，画像全体も（幅$\times$高さ$\times$チャンネル数）次元の空間上の1点であると考えることができる．今，幅と高さが両方512ピクセルだとしてみると，一枚の画像は$512 \times 512 \times 3 = 786432$という非常に高次元の空間上の1点であるということになる．しかし，自然画像はこの高次元空間上にまんべんなく分布しているわけではなく，ほとんどの場所には何のデータも存在しない．例えば，この高次元空間からランダムに1つの点を抽出してきて，それを$512 \times 512$サイズの3チャンネルカラー画像として見てみると，かなりの確率で自然な画像からは程遠い，無意味なノイズ画像が現れてくるだろうことは，想像に難くない．つまり，画像データが定義される空間全体に対して，実際に存在する画像というのはそのごく一部にしか分布していないということが予想される．

さらに，人が画像中を移動したり，カメラの向きや照明条件が変化することで現れる画像の値としての変化を考えてみると，すべての画像を表現するための空間の次元数に比べれば，それが非常に少ない次元数のパラメータによって支配されていることが想像できるのではないだろうか．例えば，カメラの位置や人の位置を示すのに十分なだけの次元数しかもたない低次元のパラメータがあれば，ある微妙に異なる画像間の本質的な変化を表現するのには十分であるかもしれない．このように，実際に観測されるデータは，データの見かけ上の次元よりもはるかに少ない次元で表される空間上に分布していると考えることができそうである．これを多様体仮説という．多様体とは，例えば球の表面がそれである．球自体は3次元空間に存在しているが，表面だけを見ると，その広がりは2次元的である．例えば，地球は3次元的に広がっているが，我々は地表面を2次元的な広がりとして捉えていることを考えてみるとよい．

さて，このような多様体仮説を考えるとき，空間の様々な位置に確率を割り当てるモデルを最尤推定で求めることには問題があることが知られている．最尤推定では，データ分布が極めて低い確率を割り当てているような領域，すなわち画像生成の場合全く自然でないような画像に対応する領域に対して，モデルが$0$より大きい確率を割り当ててしまっていたとしても，データ分布がほとんど$0$のような小さな値となっている時点でKLダイバージェンスは小さい値を取ってしまうため，この最小化によって達成される最尤推定はそのようなモデルの間違いをうまく罰することができないのである[^Arjovsky2017]．多様体仮説が成り立つようなデータの場合は，ほとんどの点において確率は$0$となっているだろうから，これは深刻な問題となる．さらに，観測データの確率分布が$0$より大きい値となっている領域（サポートという）と，モデルが$0$より大きい確率を割り当てている領域とが互いに全く重なっていないような状況になると，この二つの分布の距離を測るためのKLダイバージェンスが不定となってしまう．

このように，最尤推定を用いた高次元データの，特に多様体仮説が成り立つようなデータの生成モデル学習には，様々な問題が存在する．本稿で解説するGANは，このような最尤推定が持つ問題を回避することができる手法の一つとなっている．

GANは，最尤推定で最小化されるKLダイバージェンスとは異なる，Jensen-Shanon（JS）ダイバージェンスという距離の最小化を行っている．JSダイバージェンスはデータ分布$p_\{\rm data}$とモデルが表現する分布$p_\{\rm model}$の平均の分布$M = (p_\{\rm data} + p_\{\rm model}) / 2$を考え，これに対するデータ分布のKLダイバージェンス$KL(p_\{\rm data} \parallel M)$およびモデルが表す分布のKLダイバージェンス$KL(p_\{\rm model} \parallel M)$をそれぞれ考慮する．このため，距離の計算に用いられる分布間（$p_\{\rm data}$と$M$の組み合わせ，もしくは$p_\{\rm model}$と$M$の組み合わせ）に必ず確率が$0$より大きくなっている領域の重なりが生じるため，最尤推定のときに起こっていた問題を回避することができるのである．

## 生成モデルの重要性

GANが最尤推定の持ついくつかの問題を回避できることが分かった．しかし，GANが行うのは，どのようなデータがどのような確率で生じるのかを表す確率密度関数の直接的な推定ではなく，データの分布に近づくよう学習した分布から，サンプルを抽出することができるGeneratorを得ることである．つまり，GANによって達成されるのは$p_\{\rm model}$からサンプルを抽出できるということであり，データの分布そのものを陽に得ることではない．にも関わらず，対象のデータの分布に近い分布からサンプルを得ることができるということは，意外にも多くの場面で役に立つ．

例えば，モデルベース強化学習について考えてみよう．強化学習とは，エージェントが環境とのインタラクションを通して行動指針（ポリシーと呼ばれる）を学習する，という問題設定を指すが，特にモデルベース強化学習では，この“環境”についても，これを表現または予測するためのモデルを何らかの形で学習する．例えば，時系列データの生成モデルを使って，環境の現在の状態とエージェントの行動から，その環境の“あり得る将来の状態”を予測することができる条件付き分布を学習すれば[^Finn2016]，望ましい将来の状態を最も生じさせやすいような行動を選択するのに役立つだろう[^Finn2017]．また，生成モデルによって仮想環境を作り，これを用いてエージェントの行動ポリシーを学習することも可能となる[^David2018]．このように，モデルベース強化学習への応用では，確率密度関数を陽に得ることができないとしても，条件付き分布からサンプルを得ることができる生成モデルがあれば，十分に有用である可能性がある．

また，半教師あり学習における有用性も知られる．深層学習アルゴリズムの中には，大量のラベル付き学習データを必要とするものが多いが，一般にラベルデータの作成には多大なコストが掛かる．半教師あり学習は，必要となるラベルの数を減らすことができる方法の一つで，少ないラベル付きデータと，ラベルのない大量のデータを使うことで汎化性能を向上させる手法のことである．紙面の都合上詳細は省くが，実際にGANを用いて非常に少ないラベルデータのみを用いた半教師あり学習で汎化性能の向上が行えたという報告がある[^Salimans2016]．

他にも，生成モデル，特にGANが持つ重要な性質として，複数のモードを持つ出力を扱うことができるという点がある．現実のタスクでは，全く同一の入力に対して，複数の異なる望ましい出力が存在するという状況があり得る．このようなとき，望ましい出力とモデルの予測の間の平均二乗誤差を最小化するような形で学習を行うモデルでは，一つの入力に複数の異なる正解が対応しているケースを，うまく扱うことができない．例えば，動画中の次のフレームを前のフレームから予測するというタスクにおいて，同じフレームから複数の異なる見え方のフレームが現れるようなデータが存在すると，予測と正解の間の平均二乗誤差を最小にするような予測モデルは次のフレームとしてあり得る様々な可能性の平均をとったような，ぼやけた予測しか生み出せなくなる．これに対して，GANの考え方を用いた損失関数を最適化の際に加えて考慮すれば，よりシャープな予測フレームを生成することが可能になる[^Lotter2015]．低解像度の画像から高解像度の画像を生成する超解像技術も，同一の入力に対し出力に複数の可能性が考えられるため，同様の問題設定となる．これもGANの枠組みを応用することで結果を改善できることが示されている[^Ledig2017]．

## GANの仕組み

それでは，以上のような特徴・利点をもつGANを，どのような方法で学習することができるのか，説明を行う．GANの学習方法はしばしば鑑定士と贋作者の戦いに例えられる．贋作者は，鑑定士の目を欺けるよう本物と可能な限り見分けがつかないような贋作を生成しようと，腕を磨く．一方，鑑定士は贋作者が生成した作品と本物の作品を見分ける目を常に鍛えておき，贋作は贋作と，本物は本物と判別するのが仕事である．この贋作者をGenerator（$G$）というニューラルネットワークに，鑑定士をDiscriminator（$D$）というニューラルネットワークに置き換えて考えると，GANの学習方法は非常に直感的に理解することができる．

それでは，図~\ref\{fig:gan_architecture}を参照しながら，さっそく画像生成の場合を考えてみよう．まず，人間がカメラを使って現実のある瞬間を切り取った“本物の写真”が大量にあるとする．さて，贋作者$G$はランダムなノイズベクトル$\{\bf z}$を画像の形をしたデータ$G(\{\bf z})$に変換するニューラルネットワークである．一方，鑑定士$D$は，$G$が生成した画像もしくは本物の写真を受け取り，受け取った画像が本物の写真であると判断されれば1を，$G$が生成した贋作であると判断されれば0を出力するようなニューラルネットワークである．生成モデルを学習させる目的は，多くの場合，高性能な贋作者$G$を作ることである．しかし，我々が持っているデータは“本物の写真”そのもののみであるから，$G$が生成した画像に対して，それをどのように改善すればより本物と見分けがつきにくくなるのか，という教師信号を直接与えることはできない．一方$D$は，$G$が生成した画像と本物の写真を見分けられるようになるために，どれが贋作でどれが本物であるかについて教師信号を受け取って教師あり学習を行うことができる．なぜなら$D$に画像を入力する際，我々はどれが$G$の出力で，どれがデータセットからサンプルした本物の画像であるか知っているからである．よって，$D$は画像が入力されたときそれが贋作と本物のどちらであるかを予測し，その正解を受け取って，通常の分類問題と同様にして学習を進めていくことができる．

一方，贋作者$G$の学習は少し複雑になる．$G$は，$D$が贋作を本物と見間違えてしまう確率をできるだけ大きくしたい．このため，$D$の出力が，$G$が生成した画像に対して1を出力（＝本物と判断）してしまったときに小さな値となる，すなわちより良い評価を得るように損失関数を設計し，これを最小化するように学習を行う．このとき，$G$の最適化に用いられる損失の値は$D$の出力（予測結果）を用いて計算されることになるが，$D$はニューラルネットワークであり，その出力について入力に関する微分を計算することができるため，$G$が生成した画像の画素値に対し，それが$D$にとってより本物と紛らわしいものになるような方向の変化はどのようなものであるか，という情報を表す勾配を得ることができるのである．これを用いて誤差逆伝播法により$G$のパラメータに関する勾配を求めれば，$G$の学習が行えることが分かる．

これが基本的なGANの学習の仕組みとなる．ここで，$D$は「教師あり学習を行うことができる」と書いたが，それはこの学習の仕組みの中では“贋作”と“本物”という属性は$D$への入力ごとに自明であり，アノテーションなしに得ることができるためで，GANの学習全体は教師なしで行われているということには注意されたい．

さて，GANを用いた論文ではほぼ必ずといっていいほど頻繁に登場する，GANの学習における目的関数を表す式を用いて，再度同じことの説明を試みる．この式に見慣れておくことが，様々なGANについての文献を読むにあたり役に立つことを期待する．贋作者$G$と鑑定士$D$という2つのニューラルネットワークの学習は，以下のような目的関数に対するミニマックス・ゲームとして定式化されている．

$$
\begin\{equation}
\label\{eq:loss_function}
\min_G \max_D V(D, G) =
\mathbb\{E}_\{ \{\bf x} \sim p_\{\rm data}(\{\bf x})} \left[ \log D(\{\bf x}) \right]
+ \mathbb\{E}_\{ \{\bf z} \sim p_\{\bf z}(\{\bf z})} \left[ \log (1 - D(G(\{\bf z}))) \right].
\end\{equation}
$$

ここで，$p_\{\rm data}(\{\bf x})$はデータ分布であり，$\{\bf x}$はそこから抽出された実例である．一方，ノイズ$\{\bf z}$は事前ノイズ分布$p_\{\bf z}(\{\bf z})$から抽出されたランダムな値を持つノイズベクトルである（$p_\{\bf z}(\{\bf z})$には一般に一様分布や正規分布が用いられることが多い）．まず，$G$はノイズ$\{\bf z}$を受け取りフェイクサンプル$G(\{\bf z})$を出力する．次に，$D$は学習データセット内の実例$\{\bf x}$を受け取り，これが学習データセット由来である確率$D(\{\bf x})$を出力する．当然$D$は，これを最大にしたい．さらに$D$は，$G$が生成したフェイクサンプル$G(\{\bf z})$も受け取って，それがデータセット由来である確率$D(G(\{\bf z}))$も予測している．$D$の立場にたてば，これは贋作に騙される確率を意味するので，最小にしたい．それは，$1 - D(G(\{\bf z}))$を最大化するのと同じである．つまり，第2項は，$D$にとっては，第1項と同様最大化したいものということになる．一方，$G$の立場にたてば，$1 - D(G(\{\bf z}))$というのはフェイクサンプルが学習データセット由来である確率$D(G(\{\bf z}))$（＝贋作で騙せる確率）を1から引いたものであるから，つまり贋作がデータセット由来でない，と見破られる確率を表しており，最小化したい．

よって，全体としてはこの目的関数は学習データセット由来の実例を学習データセット由来であると，$G$が生成したフェイクサンプルは学習データセット由来でないと，正しく見分けられる度合いを表しており，$D$の視点に立てばこの式を最大化したいということになり，$D$が判別を誤るほどリアルなサンプルを生成したい$G$の視点に立てば，この式を最小化したいということになる．

ここまで説明したGANの学習方法を振り返りながら，再び図~\ref\{fig:gan_architecture}を眺めてみると，$\{\bf x}_\{\rm fake}$が前述の$G(\{\bf z})$に，$\{\bf x}_\{\rm real}$が前述の$\{\bf x}$に，それぞれ対応していることが分かる．繰り返しになってしまうが，このようにGeneratorとDiscriminatorを互いに競い合わせるようにして学習させる点がGANの学習における特徴的な部分である．

さて，式~\ref\{eq:loss_function}中の$\mathbb\{E}$は期待値計算を意味するが，これが実際にはどのようにして近似計算されるのかについては，次節のアルゴリズム~\ref\{training}を用いて解説を行う．

### GANの学習アルゴリズム



**Algorithm 1**
ミニバッチ確率的勾配降下法によるGANの学習．$k$は1イテレーション中に何回$D$のパラメータを更新するかというハイパーパラメータである．ミニバッチサイズは$m$とする．

for 学習イテレーションの数 do

    for $k$ステップ do

        $m$個のノイズ$\left\\{ \{\bf z}^\{(1)}, \{\bf z}^\{(2)}, \dots, \{\bf z}^\{(m)} \right\}$を$p_\{\bf z}(\{\bf z})$からサンプリングする

        $m$個の実例$\left\\{ \{\bf x}^\{(1)}, \{\bf x}^\{(2)}, \dots, \{\bf x}^\{(m)} \right\}$をデータセットからサンプリングする

        Discriminator（$D$）のパラメータ$\theta_d$を以下の式に従って更新

$$
\theta_d \leftarrow \theta_d + \nabla_\{\theta_d} \frac\{1}\{m} \sum_\{i=1}^m \left[ \log D(\{\bf x}^\{(i)}) + \log (1 - D(G(\{\bf z}^\{(i)}))) \right].
$$

\STATE $m$個のノイズ$\left\\{ \{\bf z}^\{(1)}, \{\bf z}^\{(2)}, \dots, \{\bf z}^\{(m)} \right\}$を$p_\{\bf z}(\{\bf z})$からサンプリングする
\STATE Generator（$G$）のパラメータ$\theta_g$を以下の式に従って更新
\STATE $$\theta_g \leftarrow \theta_g - \nabla_\{\theta_g} \frac\{1}\{m} \sum_\{i=1}^m \log(1 - D(G(\{\bf z}^\{(i)}))).$$
\ENDFOR
\end\{algorithmic}
\end\{algorithm}
$$

アルゴリズム~\ref\{training}にGANの学習ループの流れをまとめたものを示す．ここで，$D$は$G$由来のサンプルと学習データセット由来の実例を見分けるように訓練されるが，ある学習中の1イテレーションの最中に，学習データセット内の全ての実例および同数のその時点の$G$が生成したサンプル全体を用いて$D$の更新を行うのではなく，ミニバッチサイズの分だけ実例の抽出とフェイクサンプルの生成を行ったら，これを使って一旦$D$の更新を行ってしまう．さらにこれを$k$回繰り返したら，一度$D$の更新は止め，$G$の更新を1回行うようにする．これをGANの学習全体における1イテレーションとすることで，学習にかかる時間を削減する．

ここで，$G, D$それぞれのパラメータを$\theta_g, \theta_d$と書くことにすると，それぞれの更新は，式~\ref\{eq:loss_function}についてこれらのパラメータに関する勾配を求め，それを用いて勾配降下法をベースとした様々な最適化手法に従って行えばよい．しかし，学習初期の$G$は，ランダムに初期化されたパラメータを使って，ランダムな値を持つノイズベクトルから実例らしいものを生成しようとするわけであるから，到底リアリスティックなサンプルを生成するには程遠く，明らかに学習データセット内の実例と異なってしまうため，$D$にとってはあまりにも見分けるのが容易いという状況に陥る．すると，式~\ref\{eq:loss_function}中の$\log(1 - D(G(\{\bf z})))$はすぐに飽和してしまうことになり，$G$に十分な勾配が提供されなくなってしまう．そこで，$G$の学習は，アルゴリズム~\ref\{training}に示されたような式~\ref\{eq:loss_function}全体の最小化を行うのではなく，$\log D(G(\{\bf z}^\{(i)}))$を直接最大化するように実装されることが多い．つまり，$D$が実例と見間違う確率を直接最大化するわけである．

このようにして$G$の学習を進めていくと，$G$が生成したデータ$G(\{\bf z})$が従う確率分布$p_\{\rm model}(\{\bf x})$が，学習データセット内の実例$\{\bf x}$の確率分布$p_\{\rm data}(\{\bf x})$に収束していくということが示されている[^GAN]．

\section\{GANを用いた画像生成}

ここまでで解説を行ったGANの学習方法を用いれば，基本的には$D$に“本物”として入力するデータを好きに用意することで様々なデータの生成モデルを訓練することができる．しかし，高いクオリティでの生成を行うためには，データの性質に合わせたネットワークアーキテクチャの設計が重要となる．GANの提案論文[^GAN]の中でも既に学習データセットとしてMNIST[^MNIST]やTFD[^TFD]のような画像データセットを用いた画像生成の例は紹介されている（図~\ref\{fig:original_GAN_results}）が，この論文中では$G$や$D$に使用されるニューラルネットワークがシンプルな多層パーセプトロンであった．そこで，よりリアリスティックな画像の生成を目指して，画像認識の分野で大きな成果を上げているConvolutional Neural Networks（CNN）を利用して画像生成を行う手法：Deep Convolutional Generative Adversarial Networks（DCGAN）[^DCGAN]が2016年の深層学習や表現学習に関する国際会議，International Conference on Learning Representations（ICLR）にて発表された．

\subsection\{DCGANを用いた画像生成}

DCGANの主な目的はGANの枠組みの中で訓練される2つのニューラルネットワークに，画像生成に適したアーキテクチャ上の制約を与えることでよりリアリスティックな画像を生成し，それによって教師なし学習を用いた画像のよい表現（representation）の獲得が可能であることを示すことにあった．学習方法は基本的なGANの学習アルゴリズム~\ref\{training}に従うが，$G$と$D$に用いられるニューララルネットワークの構造が異なる．

まず$G$は，100次元の一様分布からサンプルされたノイズベクトルを入力として受け取って，Transposed Convolutionレイヤ（Deconvolutionレイヤと呼ばれることもあるが，Deconvolutionは通常全く異なる計算を指す言葉であるので，Transposed Convolutionと呼ぶべきとされる）を使って画像の形に変換し，段階的にその解像度を上げていくことで最終的に$64 \times 64 [\{\rm pixel}^2]$の画像を出力するCNNとなっている．

一方，$D$は画像を入力として受け取り，1次元の値を出力する画像の2クラス分類などで一般的なCNNと同様の構造をしているが，いくつか異なる工夫が加えられている．Pooling層を無くし，畳み込みカーネルを適用するストライドを大きくすることで特徴マップの縮小を行っている点や，最終層を全結合レイヤではなくGlobal Average Poolingに置き換えて出力の次元を落としている点などが，当時よく用いられていた画像分類用のCNNの構造とは異なっていた．

また，DCGANの学習を安定化させるためには，$G$の最終層の活性化関数のみTanhとし，残りはReLU[^ReLU]とする一方，$D$の全ての層では活性化関数としてLeakyReLU[^LeakyReLU]を用いる必要がある，といったことが報告された．これは，$D$から$G$へできるだけ多くの勾配情報を逆伝播することが重要であることを示唆する．

このように，生成したいデータの性質に合わせて$G$を設計するだけでなく，学習を安定的に行うためには勾配情報がいかにして$G$に伝わっていくかに注意しながら，$D$の設計にも工夫をこらさなければならないことが分かった．詳しくはDCGAN論文[^DCGAN]を参照されたい．
 
DCGANは，CNNを用いたというだけでなく，学習の安定化を図るいくつもの工夫も同時に提案することで，結果としてより複雑な画像の生成に成功しているが，この論文ではさらに，図~\ref\{fig:DCGAN_vector_arithmetic}に示されるような，生成された画像に対応するノイズベクトル$\{\bf z}$が，その画像のよい表現（representation）となっており，画像の持つ特徴同士が合成されたような画像の生成が，この$\{\bf z}$同士の演算によって行えることなどが実演された．加えて，二つの$\{\bf z}$の間を補間することで，生成される画像の間を自然につなぐような中間的な画像を生成できることも示されている．これは画素値に対する補間とは異なり，$\{\bf z}$の空間が持つ潜在的な意味の間を補間するため，画素値間の補間よりも画像から受け取ることができる意味というより抽象的なレベルにおける補間を行うことができるということである．ぜひ論文[^DCGAN]中のFigure 8を参照されたい．

\subsection\{学習の安定化}

DCGAN登場以降，より高解像度で大規模なデータセットを用いた画像生成を目指して，これを発展させるべく多くの研究が行われたが，途中で学習が進まなくなったり，Generatorが多様な画像を生成しなくなるmode collapseと呼ばれる現象が起こるなど，GANの学習の安定化という部分には大きな課題があった．また，学習中にDiscriminatorのロスカーブを観察しても，そこから有用な情報を得るのが難しく，学習がうまくいっているのか失敗しているのか，判断できないという問題があった．

そこで，Wasserstein GAN（WGAN）[^WGAN]という新しい学習方法が提案された．\ref\{sec:generative_models}章にてGANはJSダイバージェンスの最小化を行っていると述べたが，データの分布とモデルが表現する分布の間の距離をWasserstein距離によって表し，この最小化を行うように改良したのがWGANである．これによって学習は比較的安定するようになり，またDiscriminatorのロスが分布間のWasserstein距離を近似的に表すようになったためこれを学習の成否の参考にすることが可能となった．しかし，WGANは$D$がリプシッツであるようパラメータを一定の値$c$で$[-c, c]$の範囲にクリップする必要があるなど，新たなハイパーパラメータを導入してしまうことともなった．weight clippingにより$D$にリプシッツを強制することは“clearly terrible way”であると著者も言っており[^WGAN]，その後この論文の著者らによって$D$の勾配ノルムが大きくならないようペナルティ項を損失関数に加えることでweight clippingの必要性を取り除き，かつより安定したWGANの学習を行うことができるWGAN-GP[^WGAN-GP]という手法が提案されている．WGAN-GPは，その後の多くの手法で基本形として用いられている．

一方，WGAN-GPとは異なり，$D$が1-リプシッツとなるよう各レイヤーの重みをそのスペクトルノルムを使って正規化することで，weight clippingも勾配ノルムへのペナルティ付与もせずに学習の安定化と高速化がはかれるというSpectral Normalization for Generative Adversarial Networks（SNGAN）[^SNGAN]という手法も登場した．これによって，WGAN-GPよりもさらに安定的かつ高いクオリティでの画像生成ができることが示されている[^SNGAN]．

\subsection\{高解像度画像の生成}

DCGANよりもさらに高解像度の画像生成を行うために，WGAN-GPをベースに，学習中，段階的に生成する画像の大きさを大きくしていくなどいくつかの新たな工夫を加えることで，人間の目にも本物と見間違うほどに高いクオリティの高解像度画像の生成を行うProgressive Growing of GANs（PGGAN）[^PGGAN]という手法が提案されている．PGGAN論文中では，CelebA[^Liu2015]という人の高解像度顔画像を集めたデータセットを用いて，$1024 \times 1024$という大きさの画像の生成まで可能となったことが示されている（図~\ref\{fig:PGGAN}）．その後，BigGAN[^Brock2018]という手法では，大量の計算資源を活用し，学習時のミニバッチサイズを2048個にまで増やすことで，ImageNetのようなより多様な画像が含まれるデータセットを用いても，クラス情報を条件として与えるConditional GANと呼ばれる枠組みの中で非常に高いクオリティで高解像度の画像生成（$512 \times 512$サイズまで実験が行われている）が可能であることが示された（図~\ref\{fig:biggan}）．

\subsection\{画像間変換}

ここまで紹介したものは，主にランダムノイズからデータを生成する手法であったが，Generatorに対して，ノイズだけでなく，何らかの意味のある入力を同時に与え，その入力に条件付けられた分布からのサンプルを生成するようGeneratorを訓練するConditional GAN[^Mirza2014]と呼ばれる手法も提案されている．

Conditional GANを応用し，異なるドメインの画像間の変換をGeneratorによって学習するPix2Pix[^Phillip2016]では，例えば，スケッチ画のような線画を入力として与え，それに対応したリアリスティックな画像を出力する（図~\ref\{fig:pix2pix}）など，生成される画像をGeneratorに与える入力によって操作する方法が提案された．Pix2PixにおけるConditional GANの学習方法の概略図を図~\ref\{fig:pix2pix_architecture}に示す．Pix2Pixでは，条件として画像とランダムノイズの両方を与えて画像を生成するため，同一の画像を与えた場合でも，異なるノイズを入力することで異なる結果を出力することが可能である．つまり，同一の入力に複数の望ましい出力が対応するようなケースをカバーできている．このような変換が学習できることがGANの強みの一つである．このPix2Pixは，入力の画像の構造を保持したままテクスチャや表面的なアピアランスを変更するような変換を行うことを目指し，U-Net\cite\{Ronneberger2015}というSemantic segmentationタスクに向けて提案されたCNNアーキテクチャをGeneratorに採用している．このように目的に合わせて適したニューラルネットワークの構造を選ぶことがGANの発展における鍵となったケースはしばしば見られる．

さらに，Pix2Pixを高解像度な画像生成のために発展させた手法であるPix2PixHD[^Wang2018]では，図~\ref\{fig:pix2pixhd}に示したように，Semantic segmentationタスク向けのデータセットで見られるようなラベル画像を条件としてGeneratorに入力し，それに対応したリアリスティックな画像を，$2048 \times 1024$という高解像度にて生成することに成功している．Semantic segmentationを行うCNNと逆のことをするわけである．ただし，一つのラベル画像から，複数のあり得るリアリスティックな画像を生成できる点が単純な逆変換とは異なる．このような手法は，少ないラベルデータのみが含まれるデータセットを元に，多くの学習用ラベル付きデータを合成することができる可能性を示しており，もし合成されたラベル付きデータを用いた教師あり学習によって，合成データを用いない場合より高い性能が達成できることなどが分かれば，特にアノテーションコストの高いSemantic segmentationのようなタスクにおいて大きなコスト削減につながるだろう．

\section\{おわりに}

本稿では，生成モデルの重要性に始まり，近年大きな注目を集めるGANの基本的な学習の仕組みと，GANを使った画像生成の研究が大きな注目を集めるきっかけとなったDCGAN，およびその発展手法について解説した．他にもニューラルネットワークを用いた深層生成モデルとしてはVariational Autoencoder（VAE）[^Kingma2013]という手法も注目されているが，本稿では紙面の関係上一切の説明を省いている．しかし，目的によってはGANよりもVAEを用いる方が適しているケースも存在する．また，VAEとGANの接続を試みる論文[^Rosca2017]も出てきていることから，深層生成モデルに興味を持たれた方には，VAEについてもぜひ調べてみることをおすすめしたい．

最後に，本稿で紹介したGANとその発展手法の実装の多くが，Chainer-GAN-lib\footnote\{https://github.com/pfnet-research/chainer-gan-lib}にて入手可能である．これらの実装は，深層学習フレームワークであるChainer[^Chainer]を用いてPythonで書かれており，手軽に様々な手法を用いた画像生成を試すことができるようになっている．特に画像生成の手法は，一度自らの手で実装を動かしてみることで理解が進む部分も多くあるだろうから，次の一歩として，ともかく自分のコンピュータを使ってこうした実装を動かし，画像の生成を行ってみるということをおすすめしたい．

[^feynman]: 1988年にRichard Feynmanが亡くなったときの彼の黒板より（スティーブン・ホーキングの著書 "The Universe in a Nutshell" によれば）

[^GAN]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems 27, pp. 2672–2680. Curran Associates, Inc., 2014.

[^Wang2018]: Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-resolution image synthesis and semantic manipulation with condi- tional gans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[^Tokui2015]: Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next- generation open source framework for deep learning. In Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS), 2015.

[^Salimans2016]: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Proceedings of the 30th Interna- tional Conference on Neural Information Processing Systems, NIPS’16, pp. 2234–2242, USA, 2016. Curran Associates Inc.

[^Rosca2017]: Mihaela Rosca, Balaji Lakshminarayanan, David Warde-Farley, and Shakir Mohamed. Variational approaches for auto-encoding generative adversarial networks. arXiv preprint arXiv:1706.04987, 2017.

[^Ronneberger2015]: O. Ronneberger, P.Fischer, and T. Brox. U-net: Convolutional networks for biomed- ical image segmentation. In Medical Image Computing and Computer-Assisted Inter- vention (MICCAI), Vol. 9351 of LNCS, pp. 234–241. Springer, 2015. (available on arXiv:1505.04597 [cs.CV]).

[^Radford2016]: Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learn- ing with deep convolutional generative adversarial networks. International Conference on Learning Representations, 2016.

[^Miyato2018]: Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.

[^Mirza2014]: Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, Vol. abs/1411.1784, , 2014.

[^Lotter2015]: William Lotter, Gabriel Kreiman, and David D. Cox. Unsupervised learning of visual structure using predictive generative networks. CoRR, Vol. abs/1511.06380, , 2015.

[^Liu2015]: Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015.

[^Ha2018]: David Ha and Ju ̈rgen Schmidhuber. World models. CoRR, Vol. abs/1803.10122, , 2018.

[^Isola2016]: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image trans- lation with conditional adversarial networks. arxiv, 2016.

[^Hinton2010]: Geoffrey E. Hinton Joshua Susskind, Adam Anderson. The toronto face dataset. Technical report UTML TR 2010-001, U. Toronto, 2010.

[^Karras2018]: T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive Growing of GANs for Improved Quality, Stability, and Variation. International Conference on Learning Representations, 2018.

[^Kingma2013]: Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[^Lecun1998]: Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, Vol. 86, No. 11, pp. 2278–2324, Nov 1998.

[^Ledig2017]: Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew P. Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. In 2017 IEEE Confer- ence on Computer Vision and Pattern Recognition (CVPR), pp. 105–114, July 2017.

[^Brock2018]: Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. CoRR, Vol. abs/1809.11096, , 2018.

[^Finn2017]: C. Finn and S. Levine. Deep visual foresight for planning robot motion. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pp. 2786–2793, May 2017.

[^Finn2016]: Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In Proceedings of the 30th International Con- ference on Neural Information Processing Systems, NIPS’16, pp. 64–72, USA, 2016. Curran Associates Inc.

[^Glorot2011]: Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, Vol. 15 of Proceedings of Machine Learning Research, pp. 315–323, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR.

[^Goodfellow2014]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems 27, pp. 2672–2680. Curran Asso- ciates, Inc., 2014.

[^Gulrajani2017]: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems 30, pp. 5767–5777. Curran Associates, Inc., 2017.

[^Arjovsky2017]: Martin Arjovsky, Soumith Chintala, and L ́eon Bottou. Wasserstein generative adver- sarial networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, Vol. 70 of Proceedings of Machine Learning Research, pp. 214–223, International Convention Centre, Sydney, Australia, 06–11 Aug 2017. PMLR.

[^pix2pixhd]: Nvidia/pix2pixhd. https://github.com/NVIDIA/pix2pixHD. Accessed: 2018-11-03.

[^Ng2013]: Andrew Y. Ng Andrew L. Maas, Awni Y. Hannun. Rectifier nonlinearities improve neural network acoustic models. In Proceedings of ICML Workshop on Deep Learning for Audio, Speech and Language Processing, 2013.

[^Arjovsky2017]: Martin Arjovsky and L ́eon Bottou. Towards principled methods for training generative adversarial networks. arXiv preprint arXiv:1701.04862, 2017.
