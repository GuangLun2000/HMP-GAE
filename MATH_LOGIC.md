# HMP-GNN 数学逻辑规范(Math Logic Specification)

> 本文档是代码库全部核心数学逻辑的权威总结,供论文写作 agent 使用。
> 每个公式均标注实现位置(文件:行为准),**以代码实际实现为准**,与常见论文写法不同处均已显式标注。
> 记号约定:$N$ = 客户端数,$d$ = 可训练参数展平维度(LoRA-only),$t$ = 联邦轮次(0-indexed)。

---

## 0. 总览:一轮联邦学习的完整流水线

每轮 $t$ 服务器执行(`server.py: run_round`):

1. **广播**:全局参数 $w_t$ 下发所有客户端;
2. **本地训练**:每个客户端 $i$ 做 FedProx 本地训练,上传更新 $\Delta_i^t = w_i^{local} - w_t \in \mathbb{R}^d$;
3. **攻击伪装**(Hallucination 攻击为恒等映射,见 §2);
4. **(可选)语义探针前向**:当 `semantic_weight > 0`,服务器在固定 probe 集上前向每个客户端的本地模型,得到 $P \in \mathbb{R}^{N \times K \times C}$;
5. **HMP-GAE 防御聚合**:计算信任权重 $\alpha \in \Delta^{N-1}$(单纯形),聚合 $\Delta_g = \sum_i \alpha_i \Delta_i$;
6. **全局更新**:$w_{t+1} = w_t + \eta_{server} \cdot \Delta_g$($\eta_{server} = 1.0$);
7. **评估**:全局 Clean Accuracy、Global Loss、CSE;每客户端 local acc / local CSE。

FL 全部结束后一次性计算 PPL(§7.3)。

---

## 1. 客户端本地训练(FedProx)

`client.py: BenignClient.local_train` (client.py:138-207)

客户端 $i$ 以全局模型 $w_t$ 为起点,最小化

$$
\min_{w} \; F_i(w) + \frac{\mu}{2}\,\lVert w - w_t \rVert^2
$$

- $F_i(w)$:本地数据上的交叉熵损失(SeqCLS 分类头);
- $\mu$ = `config['alpha']`,**当前默认 $\mu = 0$**,即退化为标准 FedAvg 本地步;
- 优化器 Adam(lr = `client_lr` = 5e-5),梯度裁剪 $\lVert g \rVert \le 1.0$,`local_epochs = 1`;
- 上传量:$\Delta_i^t = w_i^{local} - w_t$(仅 LoRA 可训练参数展平,CPU tensor)。

**多模态模拟说明(V1)**:不接真实多模态 encoder,用小型 decoder-only LLM + LoRA($r=8, \alpha_{LoRA}=16$)模拟多模态联邦 LLM 客户端(当前 backbone 见 §8 权威配置)。

---

## 2. 威胁模型:Hallucination 攻击(label-flip)

`attack/hallucination.py`

### 2.1 攻击定义

攻击者**不伪造更新**,而是在本地训练时对标签做翻转,然后执行与 benign 完全相同的 FedProx 训练:

$$
\Delta_a^t = \text{FedProxTrain}\big(w_t;\ \{(x_j, \tilde{y}_j)\}_{j \in \mathcal{D}_a}\big)
$$

其中翻转标签

$$
\tilde{y}_j = \begin{cases}
\text{flip}(y_j) & \text{以概率 } \rho^t \text{ (per-sample Bernoulli mask)} \\
y_j & \text{否则}
\end{cases}
$$

三种 flip 模式(`FlippedLabelDataset._apply_flip`, attack/hallucination.py:108-121):

| 模式 | $\text{flip}(y)$ |
|---|---|
| `pairwise` | 固定双射 $\pi(y)$(如 $0{\leftrightarrow}1, 2{\leftrightarrow}3, \dots$) |
| `targeted` | 恒定目标类 $c^\*$ |
| `random`(**当前默认**) | 均匀采样自 $\{0,\dots,C-1\}\setminus\{y\}$ |

### 2.2 Per-round 随机化(当前默认实验)

`prepare_for_round` (attack/hallucination.py:260-291):每轮重采样

- 翻转比例 $\rho^t \sim \mathcal{U}[0.3,\ 0.8]$(`hallu_flip_ratio_range`);
- 翻转掩码与随机目标类,种子 $\text{seed}^t = s_a \cdot 100003 + t$($s_a$ 为攻击者专属种子,大素数避免多攻击者碰撞)。

效果:攻击梯度方向逐轮非平稳(CSE / local acc 震荡),比冻结的错误标签流形更接近现实攻击者。

### 2.3 隐蔽性(stealth)论证

由于攻击者跑的是**真实的本地 SGD**(只是标签被污染),更新的几何统计量(norm、cosine)自然落在 benign 分布附近,即约束

$$
\lVert \omega_a - \omega'_a \rVert \le \epsilon
$$

**天然满足**,不需要显式投影。`camouflage_update` 是恒等映射(attack/hallucination.py:311-317)。这正是纯几何防御(Krum/norm-clipping 等)难以检测该攻击的原因,也是引入语义信号(§5.3 Signal 3)的动机。

**与经典 Byzantine baseline 的关键区别**:`sign_flipping`/`gaussian`/`alie` 是 dataset-free 的(不读本地数据、直接伪造更新);Hallucination 攻击者**使用**自己的本地数据。

---

## 3. HMP-GAE 防御:结构总览

`hmp_gae/runtime.py: HMPGAERuntime.aggregate`

V1–V7 的基础 HMP-GAE 每轮在服务器端执行(全部在 CPU,因 $N$ 小):

```
Δ (N×d)  ──η_i 特征提取──▶  η (N×64)
          ──k-NN 超图──▶    H, D_V, D_E   (N×N)
          ──HMP encoder──▶  Z (N×32)
          ──GAE decoder──▶  Â = σ(ZZᵀ),  Ĥ = σ(Z W_decᵀ)
          ──自监督训练 5 步 (Adam)──
          ──四信号信任打分──▶ s ∈ ℝᴺ
          ──suspicion + EMA + sigmoid gate──▶ α ∈ Δ^{N-1}
          ──加权聚合──▶ Δ_g = Σᵢ αᵢ Δᵢ
          ──EMA 更新 Z_hist──
```

encoder/decoder 参数与 Adam 状态**跨轮持久化**(不重置),即 GAE 随 FL 轮次持续在线学习。
当前 V8 保留节点特征、HMP 与跨轮状态，但替换了超图固定方式、decoder
目标和最终决策协作方式；完整定义见 §6.6。不要用上图的“每 step 重建 H +
scalar gate”描述 V8。

---

## 4. HMP-GAE 各阶段数学定义

### 4.1 节点特征 $\eta_i$

`hmp_gae/node_features.py`

$$
\eta_i = f_{enc}\big(\underbrace{R\,\Delta_i}_{\text{JL 投影}},\ \underbrace{c_i}_{\text{上下文统计}},\ \underbrace{h_i^{t-1}}_{\text{历史嵌入}}\big) \in \mathbb{R}^{64}
$$

**(a) 固定随机投影(JL embedding)**(node_features.py:64-94):
$R \in \mathbb{R}^{d \times 64}$,$R_{jk} \sim \mathcal{N}(0, 1/64)$,固定种子 42,**跨轮不变、不训练**——保证特征几何在时间上稳定(否则历史嵌入会活在漂移的基底里)。

**(b) 上下文统计** $c_i \in \mathbb{R}^4$(node_features.py:29-57):

$$
c_i = \big[\ \log(1 + \lVert\Delta_i\rVert_2),\ \ \text{mean}(\Delta_i),\ \ \text{std}(\Delta_i),\ \ \cos(\Delta_i, \bar{\Delta})\ \big], \quad \bar{\Delta} = \tfrac{1}{N}\sum_j \Delta_j
$$

**(c) 历史嵌入** $h_i^{t-1} = z^{hist}_i \in \mathbb{R}^{32}$(EMA 缓存,见 §4.6;冷启动为零向量)。

**(d) $f_{enc}$**:2 层 MLP,$\text{Linear}(64{+}4{+}32 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 64)$,可训练。

### 4.2 k-NN 超图构建

`hmp_gae/hypergraph.py: knn_hypergraph`

每个节点 $i$ 生成一条以自身为中心的超边:

$$
\varepsilon_i = \{i\} \cup \text{top-}k\big(\ \cos(\eta_i, \eta_j),\ j \ne i\ \big), \qquad k_{eff} = \min(k, N{-}1)
$$

- **$M = N$**(每节点一条超边),使关联矩阵 $H \in \{0,1\}^{N \times N}$ 是方阵、decoder 维度跨轮稳定;
- $H_{i,e} = 1 \iff i \in \varepsilon_e$(列 $e$ 以节点 $e$ 为中心);
- 度:节点度 $d_V(i) = \sum_e H_{i,e}$,超边度 $d_E(e) = \sum_i H_{i,e} = k_{eff} + 1$(含自身);
- 度逆以向量形式保存:$D_V^{-1},\ D_E^{-1}$。

**默认 $k = 2$($N=7$ 时)**:更大的 $k$ 会强迫 benign 节点把 attacker 纳入自己的超边,稀释孤立信号;$k=2$ 使 2-attacker 子簇更紧、graph_residual 对比更锐(main.py config 注释)。

### 4.3 HMP Encoder(两阶段超图消息传递)

`hmp_gae/encoder.py`(对应论文 eq. 15/16)

每层 $l$ 先 node→hyperedge,再 hyperedge→node:

$$
E^{(l)} = \sigma\big(D_E^{-1}\, H^\top\, Z^{(l)}\, W_E^{(l)}\big), \qquad
Z^{(l+1)} = \sigma\big(D_V^{-1}\, H\, E^{(l)}\, W_V^{(l)}\big)
$$

- $\sigma = \text{ReLU}$;$Z^{(0)} = \eta$;
- 实现上 $W_E$ 先作用于 $Z$ 再乘 $H^\top$(数学等价);dropout 施加在 $E^{(l)}$ 上(默认 0);
- $L = 2$ 层,维度 $64 \to 64 \to 32$,输出 $Z \in \mathbb{R}^{N \times 32}$。

### 4.4 GAE Decoder

`hmp_gae/decoder.py`(对应论文 eq. 17/18)

**成对邻接(内积 decoder)**:
$$
\hat{A}_{ij} = \sigma(z_i^\top z_j), \qquad \hat{A} = \sigma(Z Z^\top) \in [0,1]^{N \times N}
$$

**超边关联(线性投影 decoder)**:
$$
\hat{H}_{i,e} = \sigma(z_i^\top w^{dec}_e), \qquad \hat{H} = \sigma(Z\, W_{dec}^\top) \in [0,1]^{N \times M},\ M = N
$$

$W_{dec} \in \mathbb{R}^{N \times 32}$ 可训练。BCE 用 logits 版本以保证数值稳定。

### 4.5 自监督损失(对应论文 eq. 21 + 历史正则)

`hmp_gae/losses.py: total_loss`

$$
\mathcal{L} = \lambda_H\, \mathcal{L}_{rec}^H \;+\; \lambda_A\, \mathcal{L}_{smooth} \;+\; \lambda_{hist}\, \mathcal{L}_{hist} \;+\; \lambda_{wd}\, \lVert\theta\rVert_2^2
$$

**(a) 超边重构 BCE**(带类不平衡加权,losses.py:34-49):

$$
\mathcal{L}_{rec}^H = \text{BCE}_{w^+}\big(H,\ \hat{H}_{logits}\big), \qquad
w^+ = \text{clamp}\Big(\frac{\#\{H{=}0\}}{\#\{H{=}1\}},\ 1,\ 10\Big)
$$

**(b) Laplacian 平滑项**(losses.py:52-66;注意:论文原 $\mathcal{L}_B(Z)$ 项实为平滑项而非 BCE,代码已按此重命名):

$$
\mathcal{L}_{smooth} = \frac{1}{N^2} \sum_{i,j} \hat{A}_{ij}\, \lVert z_i - z_j \rVert_2^2
\;\;\Big(= \frac{2}{N^2}\,\text{tr}(Z^\top L Z),\ L = D - \hat{A}\Big)
$$

**(c) 历史一致性**(losses.py:69-80,$Z^{hist}$ detach,冷启动时为 0):

$$
\mathcal{L}_{hist} = \frac{1}{N \cdot d_z} \lVert Z - Z^{hist} \rVert_F^2
$$

**V1–V7 训练细节**:$\lambda_H = \lambda_A = 1.0$,$\lambda_{hist} = 0.5$,$\lambda_{wd} = 10^{-5}$;每轮 Adam(lr $10^{-3}$)5 步;梯度裁剪 max-norm 5.0;超图 $H$ 每步随 $\eta$ 重建(即 $H$ 依赖当前 $f_{enc}$ 参数)。训练完成后以 eval 模式重新前向一次得到用于打分的 $\eta, H, Z, \hat{A}$。V8 不采用这个自指训练目标，见 §6.6。

### 4.6 历史嵌入 EMA($Z^{hist}$)

`runtime.py: _update_history`

$$
z^{hist}_i \leftarrow \beta_h\, z^{hist}_i + (1 - \beta_h)\, z_i^t, \qquad \beta_h = 0.9
$$

首次观测直接初始化 $z^{hist}_i = z_i^t$。$Z^{hist}$ 同时进入:①节点特征的 $h_i^{t-1}$ 输入;②$\mathcal{L}_{hist}$;③Signal 4 的 hist_dev。

---

## 5. 信任打分(四信号)

`hmp_gae/trust_scorer.py: compute_trust_weights`

### 5.1 Signal 1 — graph_residual(超图孤立度,主信号)

基于**确定性**的 k-NN 超图 $H$(给定 $\eta$ 后构建是确定性的;$\eta$ 出自**当轮**自监督训练后的 node encoder $f_{enc}$——见 §4 训练细节,$H$ 随 $f_{enc}$ 参数变化——但不需要任何跨轮 warmup 或收敛的 GAE decoder,round 0 即可用):

$$
\text{reach}_i = \#\big\{\, j \ne i : (HH^\top)_{ij} > 0 \,\big\}, \qquad
r^{graph}_i = 1 - \frac{\text{reach}_i}{N - 1} \in [0, 1]
$$

直觉:attacker 更新彼此相似、形成紧凑子簇,与 benign 多数共享的超边少 → reach 低 → residual 高。

### 5.2 Signal 2 — recon_residual(GAE 重构孤立度)

基于学到的 $\hat{A}$(encoder 训练收敛后逐渐变锐利):

$$
r^{recon}_i = 1 - \frac{1}{N-1} \sum_{j \ne i} \hat{A}_{ij}
$$

### 5.3 Signal 3 — sem_div(语义散度,行为指纹)

`trust_scorer.py: _semantic_divergence_signal`

**探针来源**:服务器持有固定 probe 集($K$ 条,当前 $K = 100$,类分层采样 `semantic_probe_stratified=True`——标签**只用于平衡采样,绝不进入打分**,信号保持 label-free)。每轮把每个客户端的本地模型($w_t + \Delta_i$)在 probe 集上前向,得 per-sample softmax $P_i^k \in \Delta^{C-1}$。

**参考分布两种模式**:

- `pairwise`(legacy):对每个有序对做对称 KL 再平均

$$
\text{div}_i = \frac{1}{(N-1)K} \sum_{j \ne i} \sum_{k=1}^{K} \tfrac{1}{2}\Big[\text{KL}(P_i^k \,\|\, P_j^k) + \text{KL}(P_j^k \,\|\, P_i^k)\Big]
$$

  弱点(非 IID):合法异质的 benign 也会偏离 peers 而被罚;且每个 benign 分数被其到 attacker 的距离抬高(压缩 attacker/benign 对比度)。

- `median`(**当前默认,推荐**):与逐样本逐类中位数共识比较

$$
R^k = \text{normalize}\big(\text{median}_i\, P_i^k\big), \qquad
\text{div}_i = \frac{1}{K}\sum_{k=1}^{K} \tfrac{1}{2}\Big[\text{KL}(P_i^k \| R^k) + \text{KL}(R^k \| P_i^k)\Big]
$$

  attacker 是少数派、无法移动中位数:benign 分数只剩自身异质性偏差,attacker 分数度量其系统性错误——对比度更大,对 <50% attacker 比例鲁棒。

- 可选 `semantic_confidence_weight`(当前 False):按样本以 $w_{ik} = \max_c P_{ikc}$ 加权——"自信地错"(attacker)全额计入,"不自信地不同"(典型非 IID benign)被折扣。

直觉:honest 客户端在每条样本上收敛于同一正确类 → 散度低;label-flip attacker 的 per-sample 预测被系统性反转 → 散度高,**即使其更新在几何上完全 stealth 也藏不住**——这是与 Signal 1/2(纯更新几何)正交的输出行为信号。

### 5.4 Signal 4 — hist_dev(历史偏离,默认权重 0)

$$
r^{hist}_i = \lVert z_i - z^{hist}_i \rVert_2
$$

**默认关闭($\beta = 0$)**:实验(Y5)发现 benign 客户端在真实学习中的漂移**大于**被困在固定错误标签流形上的 attacker,信号方向在稳态会反转(warmup 期 R1-R11 方向正确 100%,稳态 R26+ 仅 28%)。保留 `hist_warmup_rounds` 相位门控:设为整数 $T_w$ 时仅在 $t < T_w$ 启用,$\beta_{eff} = \beta \cdot \mathbb{1}[t < T_w]$。

### 5.5 鲁棒标准化(z-score)

`trust_scorer.py: _zscore`

每个信号跨客户端标准化后再融合(尺度不变性):

- `std` 模式(legacy):$z = (x - \text{mean}) / \text{std}$。缺陷:attacker 比例升高时会把 mean/std 拉向自己、自身 z 缩回 benign 区间。
- `mad` 模式(**当前默认**):

$$
z = \frac{x - \text{median}(x)}{1.4826 \cdot \text{MAD}(x)}, \qquad \text{MAD} = \text{median}\,|x - \text{median}(x)|
$$

  median/MAD 由 benign 多数决定,attacker z 在 <50% 比例下保持大。MAD 退化为 0 时(如 graph_residual 粗量化、benign 多数完全并列)回退到 std 尺度,避免单量子偏差爆到 $z \sim 10^5$。

- 截断:$z \leftarrow \text{clip}(z, -c, c)$,**当前 $c = 10$**(`zscore_clip`),防单个极端离群点主导加权和。

### 5.6 组合信任 logit 与 softmax 权重

$$
s_i = -\Big( w_g\, z(r^{graph}_i) + w_r\, z(r^{recon}_i) + w_s\, z(\text{div}_i) + \beta_{eff}\, z(r^{hist}_i) \Big)
$$

**当前权重**:$w_g = 1.0$(主)、$w_r = 0.3$(次)、$w_s = 1.0$(语义)、$\beta = 0$。高 $s$ = 可信。

诊断用 softmax 权重(`trust_mode='softmax'` 时直接采用):

$$
\alpha^{sm}_i = \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}, \qquad \tau = 0.1
$$

$\tau \to 0$ 退化为 Krum 式硬选择;$\tau \in [0.05, 0.5]$ 为软拒绝。缺陷:benign residual 几乎并列时权重会集中到 1-2 个客户端,浪费协作收益——因此生产路径用 §6 的 reject-then-FedAvg。

---

## 6. 从信任到聚合权重(gating)

`trust_scorer.py: _suspicion_signal / gate_diagnostics / reject_soft_weighted`(gate 的应用在 `runtime.py` 的 soft-reject 分支)

### 6.1 怀疑分数(suspicion)

`gate_signal` 选择驱动门控的信号:

- `'graph'`:$\text{sus}_i = z(r^{graph}_i)$(仅主信号,向后兼容);
- `'combined'`(**当前默认,semantic_weight>0 时自动升级,亦已显式配置**):$\text{sus}$ 源于 $-s$,两种归一化:
  - `gate_rezscore=True`(legacy 双重 z-score):$\text{sus}_i = z(-s_i)$。缺陷("替罪羊税"):强行把每轮压到 ±σ 尺度,**全 benign 轮也必然把最极端者推过阈值**,每个干净轮都削 benign 权重;
  - `gate_rezscore=False`(**当前默认,推荐**):

$$
\text{sus}_i = \frac{-s_i}{\lVert w \rVert_2}, \qquad
\lVert w \rVert_2 = \sqrt{\textstyle\sum_{k \in \text{active}} w_k^2}
$$

    各分量已 z-scored,加权和携带绝对尺度:全 benign 轮 sus ≈ 0、无人被门控,attacker 落在 $|\text{sus}| \gg \theta$。除以**活跃**信号权重的 L2 范数使阈值以"每信号 z 单位"表达,对权重缩放和信号开关不变(当前活跃 $\lVert w\rVert_2 = \sqrt{1^2 + 0.3^2 + 1^2} \approx 1.449$)。

### 6.2 跨轮 suspicion EMA

`runtime.py: _smooth_suspicion`

$$
\widetilde{\text{sus}}_i^t = \beta_s\, \widetilde{\text{sus}}_i^{t-1} + (1 - \beta_s)\, \text{sus}_i^t, \qquad \beta_s = 0.6
$$

直觉:benign 轮流坐"本轮最极端"的位子,其 EMA 回归 ~0;attacker 每轮都可疑,EMA 保持高位。代价:检测滞后约 $1/(1-\beta_s) \approx 2\text{-}3$ 轮。首次观测直接初始化。

### 6.3 Sigmoid 软拒绝门 + 数据量 FedAvg

`trust_scorer.py: reject_soft_weighted`

$$
g_i = \sigma\big(-\kappa\, (\widetilde{\text{sus}}_i - \theta)\big), \qquad
\alpha_i = \frac{D_i\, g_i}{\sum_j D_j\, g_j}
$$

- $\kappa = 2.0$(steepness;$\kappa{=}1$ 很平滑,$\kappa{=}3$ 近二值);
- $\theta$ = `reject_z_threshold`:**gate_rezscore=False 时为 2.5**(per-signal robust-z 单位);gate_rezscore=True 时用 0.75——**两个键必须一起改**;
- $D_i$ = 客户端数据量(benign 用真实 `len(data_indices)`,attacker 用 `claimed_data_size`);
- 语义:$\text{sus} \ll \theta \Rightarrow g \approx 1$(全权重);$\text{sus} = \theta \Rightarrow g = 0.5$;$\text{sus} \gg \theta \Rightarrow g \approx 0$。

**设计理由(detection 与 weighting 解耦)**:信任信号只负责**压制** attacker;benign 之间按自然数据量权重聚合,保留协作学习收益;软门无悬崖,阈值失准时优雅退化。

**安全兜底**(keep_min):若 $\#\{i : g_i > 0.1\} < k_{min}$($k_{min}=1$),强制保留 sus 最低的 $k_{min}$ 个客户端(gate 置 1)。

另有两种备选模式:`reject_then_fedavg`(硬二值 mask $\mathbb{1}[\text{sus}_i \le \theta]$ 后数据量 FedAvg,阈值敏感)与 `softmax`(§5.6)。

### 6.4 聚合与全局更新

$$
\Delta_g = \sum_{i=1}^{N} \alpha_i\, \Delta_i, \qquad
w_{t+1} = w_t + \eta_{server}\, \Delta_g \quad (\eta_{server} = 1.0)
$$

(`trust_scorer.py: weighted_aggregate`;server.py:347-350)

### 6.5 退化与冷启动

- **$N \le 2$**:HMP 消息传递不适定,自动回退 FedAvg(defense/__init__.py:203);运行时任何异常也逐轮回退 FedAvg 并记录 `fallback_reason`;
- **冷启动**:Signal 1 只依赖原始投影更新的 k-NN 超图,**round 0 即生效**;`cold_start_fallback=False`(默认)不做首轮 FedAvg 回退;
- **CLAUDE.md 备注**:超图信号在小 $N$ 下不稳(文档写 $N \le 4$ fallback,代码实际阈值为 $N \le 2$——论文表述时以代码为准并注明小 $N$ 局限)。

### 6.6 V8：CSE 种子驱动的双视图超图传播（当前主路径）

`trust_mode='v8_hmp_cse_propagation'` 不再把超图压缩成单个孤立度阈值，
而是让 CSE 与超图承担互补角色：CSE 提供高精度异常种子，超图利用客户端
关系找出与种子同机制、但自身 CSE 尚未越过硬阈值的同伴。

**(a) 两个独立且当轮固定的视图。** 更新视图不使用可训练的 $\eta$ 构图，
而使用固定 JL 投影

$$
u_i=R\Delta_i,\qquad
\varepsilon_i^u=\{i\}\cup\operatorname{topk}_{j\ne i}\cos(u_i,u_j),
$$

得到 $H^u$。行为视图只使用共享 probe 上的 per-sample softmax
$P_i^{(q)}$，不使用 probe label：

$$
\operatorname{JS}_{ij}=\frac1K\sum_q\frac12\left[
  \operatorname{KL}(P_i^{(q)}\Vert M_{ij}^{(q)})+
  \operatorname{KL}(P_j^{(q)}\Vert M_{ij}^{(q)})
\right],\quad
M_{ij}^{(q)}=\frac{P_i^{(q)}+P_j^{(q)}}2,
$$

$$
S^b_{ij}=1-\frac{\operatorname{JS}_{ij}}{\log 2},\qquad
\varepsilon_i^b=\{i\}\cup\operatorname{topk}_{j\ne i}S^b_{ij},
$$

得到 $H^b$。令 $M^v_{ij}=1$ 当且仅当 $i,j$ 在视图 $v$ 中互相选择，
则共识关系

$$
C=M^u\odot M^b.
$$

传播超图 $H^c$ 以每个节点为中心，包含自身及 $C$ 中与其互邻的节点。
因此单一更新几何偶合或单一行为偶合都无权传播风险。

**(b) 固定拓扑上的可学习 HMP-GAE。** 当轮 5 个优化 step 都使用同一个
$H^u$；这切断了旧路径“可学习 $\eta\to H\to$ 重构同一个 $H$”的反馈环。
V8 的每层加入残差和 LayerNorm：

$$
E^{(l)}=\operatorname{ReLU}\!\left((D_E^u)^{-1}(H^u)^\top
Z^{(l)}W_E^{(l)}\right),
$$

$$
Z^{(l+1)}=\operatorname{LN}\!\left((D_V^u)^{-1}H^uE^{(l)}W_V^{(l)}
+Z^{(l)}W_{skip}^{(l)}\right).
$$

最后一层不做 ReLU，允许 signed latent。成对 decoder 改为固定尺度
$\gamma=4$ 的 cosine logit：

$$
a_{ij}=\gamma\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2},
\qquad \hat A_{ij}=\sigma(a_{ij}).
$$

这样非邻居可以得到负 logit 和 $\hat A_{ij}<0.5$；旧版 final-ReLU 加
$ZZ^\top$ 无法保证这一点。V8 的结构目标是固定的 direct-mutual update
邻接 $A^u=M^u$，而不是 decoder 自己产生的平滑权重：

$$
\mathcal L_{V8}=\lambda_H\operatorname{BCE}(H^u,\hat H)+
\lambda_A\operatorname{BCE}_{i\ne j}(A^u,a)+
\lambda_{hist}\mathcal L_{hist}+\lambda_{wd}\lVert\theta\rVert_2^2.
$$

每节点的第二项均值作为 V8 的真实 `recon_residual`。

**(c) HMP 风险传播。** 先由共识超图计算 node→edge→node 算子，删除
对角并对剩余质量按行归一：

$$
P=\operatorname{RowNorm}\!\left(\operatorname{OffDiag}\left[
(D_V^c)^{-1}H^c(D_E^c)^{-1}(H^c)^\top\right]\right).
$$

再用 GAE affinity 衰减，而**不把衰减后的质量重新归一到 1**：

$$
T=P\odot\hat A,\qquad \sum_jT_{ij}\le1.
$$

这个“次随机”约束是必要的；否则节点只有一条边时，任意弱 affinity 都会
被重新放大成 1，GAE 实际不起作用。

**(d) CSE 决策权与共享 rank cap。** V8 Stage A 与 V5 完全相同。令
$r_i=\operatorname{CSE}_i/\max(\operatorname{median}_{lower}(\operatorname{CSE}),
\epsilon)$，V5 在 top-$K_B$ rank cap 内取 $r_i>\tau$ 的集合 $F$，其成员
使用原 V5 ramp multiplier。它们作为不可被替换的种子：

$$
q_i=\sum_{j\in F}T_{ij},\qquad
e_i=\operatorname{clip}\left(\frac{r_i-1}{\tau-1},0,1\right),\qquad
J_i=q_i e_i.
$$

只有 $i\notin F$、$J_i>0$ 且 CSE flag 后仍有 rank-cap 余额的客户端，
才能按 $(J_i,r_i,-i)$ 排序进入传播集合 $G$；CSE flag 永远优先。传播
客户端的连续乘子为

$$
m_i^{prop}=1-(1-m_{floor})J_i,
$$

最终 $m_i$ 对 $F$ 取 V5 ramp、对 $G$ 取 $m_i^{prop}$、其余精确取 1，
并按数据量归一：

$$
\alpha_i=\frac{D_i m_i}{\sum_jD_jm_j}.
$$

**结构保证与可证伪条件。** 无 V5 种子、无通向种子的双视图共识路径、
无 $r_i>1$ 的同伴，或 rank cap 无余额时，代码直接返回 V5 tensor，
逐元素相同；弱 affinity 则产生同比例的轻惩罚，而不是被重新放大。
超图是否产生增量必须从 `v8_propagated_flagged`、`v8_joint_evidence`、
`v8_consensus_edge_count` 与 `v8_propagation_matrix` 判断；若全程无传播，
实验结论就是 V8 退化为 V5，不能把 CSE 的收益归因于超图。反过来，若两个
攻击者都未形成 CSE 种子，V8 也不会自行检测；这是防止几何 scapegoat 的
保守边界。该模式仍依赖服务器 full-test local CSE，且与只伪造 update、
不改变 `client.model` 的 `crafts_update` 攻击不兼容。

---

## 7. 评估指标

### 7.1 Clean Accuracy / Global Loss

标准测试集准确率与平均交叉熵,每轮一次(全局模型)。

### 7.2 CSE — Classification Semantic Entropy(每轮,免费)

对测试集每条样本的 softmax 类分布求 Shannon 熵后取均值:

$$
\text{CSE} = \frac{1}{|\mathcal{T}|} \sum_{x \in \mathcal{T}} \Big(- \sum_{c=1}^{C} p(c \mid x)\, \log p(c \mid x)\Big)
$$

- 低 CSE = 预测更自信;hallucination 攻击下模型置信度下降 → CSE 上升;
- 定位:Farquhar 式 semantic entropy 的**无生成代理**(no-generation surrogate),以 $C$ 个类标签充当 "semantic clusters";
- 与 accuracy/loss 共享同一次测试集前向(每轮免费);同时逐客户端计算 local acc / local CSE(本地模型 $w_t + \Delta_i$ 在服务器测试集上)。

### 7.3 PPL — 困惑度(FL 结束后一次性)

`evaluation_hallucination.py` + `decoder_adapters.py`

把最终 LoRA 微调的 backbone 迁移进 `AutoModelForCausalLM`(SeqCLS → CausalLM,需 `save_global_checkpoint=True`),在类分层的测试子集($n = 200$)上:

$$
\text{PPL} = \exp\Big( \frac{1}{n} \sum_{j=1}^{n} \text{NLL}_j \Big)
$$

$\text{NLL}_j$ = 第 $j$ 条样本的 shifted-label 平均 token 负对数似然(HF `outputs.loss`)。另报 per-class PPL。encoder-only backbone 优雅跳过。

---

## 8. 权威默认配置(当前实验)

来源:`main.py` 的 `main()` config dict(**唯一权威 config 源**;不标行号——main.py 常改)。**本表是撰写时快照,与 config 现值冲突时一律以 config 为准。** 当前实验名(随 config 变):`agnews-(non-iid0.5)-hmpgae-v8-hallu(localround=1,seed=42,r50,len128,flip0.3-0.8)-llama3.2-1b`。

| 组 | 参数 | 值 |
|---|---|---|
| FL | $N$ / attackers / rounds | 7 (5 benign + 2 attacker) / 2 / 50 |
| FL | client_lr / server_lr / local_epochs / FedProx $\mu$ | 5e-5 / 1.0 / 1 / 0.0 |
| 模型 | backbone / LoRA | Llama-3.2-1B + LoRA($r{=}8$, $\alpha{=}16$, dropout 0.1) |
| 数据 | dataset / 分布 / 规模 | AG News($C{=}4$)/ non-IID Dirichlet(0.5) / 10K 子集, max_length 128 |
| 攻击 | mode / $\rho^t$ / reseed | random flip / $\mathcal{U}[0.3, 0.8]$ / per-round |
| 超图 | proj_dim / eta_dim / $k$ | 64 / 64 / 2 |
| GAE | hidden / latent / $L$ / steps / lr | 64 / 32 / 2 / 5 / 1e-3 |
| 损失 | $\lambda_H$ / $\lambda_A$ / $\lambda_{hist}$ / wd | 1.0 / 1.0 / 0.5 / 1e-5 |
| 信任 | $w_g$ / $w_r$ / $w_s$ / $\beta$ | 1.0 / 0.3 / 1.0 / 0.0 |
| 语义 | reference / probe $K$ / stratified / conf-weight | median / 100(AG News 每类 25)/ True / False |
| 鲁棒 | zscore_mode / clip / gate_rezscore / $\beta_s$ | mad / 10.0 / False / 0.6 |
| 决策 | trust_mode / $\tau$ / rank cap / $m_{floor}$ / $r_{hard}$ | v8_hmp_cse_propagation / 1.85 / 2 / 0.10 / 2.5 |
| 历史 | $\beta_h$ / hist_warmup | 0.9 / None |

**Legacy(2026-07 前)复现开关**(六键同时覆盖):`{'zscore_mode':'std', 'gate_rezscore':True, 'sus_ema_beta':0.0, 'reject_z_threshold':0.75, 'semantic_reference':'pairwise', 'semantic_probe_stratified':False}`。

---

## 9. 论文写作要点(叙事逻辑备忘)

1. **V8 的互补分工**是核心论点:CSE 提供高精度种子，update/probe 双视图超图保留关系并提升漏检同伴的 recall，GAE affinity 只做连续衰减；任何单通道都不能独立扩权。
2. **detection、传播与 weighting 解耦**:CSE 决定不可替换的 seed，HMP 只使用剩余 rank cap，最后仍回归数据量 FedAvg——避免 softmax 权重集中破坏协作学习。
3. **鲁棒统计栈的动机链**:std z-score 被 attacker 污染 → MAD;双重 z-score 有"替罪羊税" → 绝对尺度 sus + 权重范数归一;单轮噪声 → 跨轮 EMA;pairwise KL 在非 IID 下压缩对比度 → median 参考。每步都有对应的失效模式与修复。
4. **hist_dev 的负结果**可以写:方向在 warmup 正确、稳态反转(benign 学真特征漂移 > attacker 困在错误标签流形)——这解释了为何 $\beta = 0$ 且保留相位门控接口。
5. **符号对照**:$\eta_i$ ↔ node_features.py;$H, D_V, D_E$ ↔ hypergraph.py;eq.15/16 ↔ encoder.py;$\hat{A}, \hat{H}$ (eq.17/18) ↔ decoder.py;eq.21 ↔ losses.py;$s, \alpha$ ↔ trust_scorer.py。
6. **版本不能混写**:V1–V7 是 final-ReLU、inner-product decoder、每 step 随 $\eta$ 重建 $H$ 和 self-smoothness；V8 是 fixed-JL topology、residual/signed HMP、cosine decoder 和 fixed-topology BCE。GAE 都跨轮在线持续训练，但机制与损失不可互相冒充。
