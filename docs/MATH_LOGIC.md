# HMP-GAE 数学逻辑规范(Math Logic Specification)

> 本文档只维护**当前实现**的数学机制、符号和代码映射。运行方式见
> [README.md](../README.md)，文档分工见 [docs/README.md](README.md)，版本选择、
> 被否决方案和实验契约见 [docs/DECISION.md](DECISION.md)。实验参数现值只读
> `main()` 的 `config`，不要从本文档推断默认值。
>
> 2026-08-11 起代码只保留三个 trust 模式:V4、V5(V8 的决策层与 matched-run
> 基线)与 V8(当前机制)。V1–V3 几何信任栈与 V6/V7 已从代码移除,其数学定义
> 只存在于 git 历史和 [docs/DECISION.md](DECISION.md) 的历史条目中,本文档不再
> 描述它们。

每个公式均标注实现位置(以文件和符号为准),与常见论文写法不同处会显式
标注。记号约定:$N$ = 客户端数,$d$ = 可训练参数展平维度(LoRA-only),
$t$ = 联邦轮次(0-indexed)。

---

## 0. 总览:一轮联邦学习的完整流水线

每轮 $t$ 服务器执行(`server.py: run_round`):

1. **广播**:全局参数 $w_t$ 下发所有客户端;
2. **本地训练**:每个客户端 $i$ 做 FedProx 本地训练,上传更新 $\Delta_i^t = w_i^{local} - w_t \in \mathbb{R}^d$;
3. **攻击伪装**(Hallucination 攻击为恒等映射,见 §2);
4. **本地 CSE 评估**(聚合前):服务器在 full-test 集上前向每个客户端的本地模型 $w_t + \Delta_i$,得逐客户端 $\operatorname{CSE}_i$(§5 的唯一检测统计量;server 强制逐轮计算,缺失即崩溃);
5. **(V8 必需)语义探针前向**:当 `semantic_weight > 0`,服务器在固定 probe 集上前向每个客户端的本地模型,得 $P \in \mathbb{R}^{N \times K \times C}$;
6. **HMP-GAE 防御聚合**:计算信任权重 $\alpha \in \Delta^{N-1}$(单纯形),聚合 $\Delta_g = \sum_i \alpha_i \Delta_i$;
7. **全局更新**:$w_{t+1} = w_t + \eta_{server} \cdot \Delta_g$($\eta_{server} = 1.0$);
8. **评估**:全局 Clean Accuracy、Global Loss、CSE;每客户端 local acc / local CSE。

FL 全部结束后一次性计算 PPL(§7.3)。

---

## 1. 客户端本地训练(FedProx)

`client.py: BenignClient.local_train`

客户端 $i$ 以全局模型 $w_t$ 为起点,最小化

$$
\min_{w} \; F_i(w) + \frac{\mu}{2}\,\lVert w - w_t \rVert^2
$$

- $F_i(w)$:本地数据上的交叉熵损失(SeqCLS 分类头);
- $\mu$ = `config['alpha']`;当 $\mu = 0$ 时退化为标准 FedAvg 本地步;
- 优化器、学习率、梯度裁剪和本地 epoch 均由代码与 `main()` config 决定;
- 上传量:$\Delta_i^t = w_i^{local} - w_t$(仅 LoRA 可训练参数展平,CPU tensor)。

**多模态模拟说明**:不接真实多模态 encoder,以 decoder-only LLM + LoRA
更新模拟多模态联邦 LLM 客户端;backbone 与 LoRA 参数由 `main()` config 决定。

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

三种 flip 模式(`attack/hallucination.py: FlippedLabelDataset._apply_flip`):

| 模式 | $\text{flip}(y)$ |
|---|---|
| `pairwise` | 固定双射 $\pi(y)$(如 $0{\leftrightarrow}1, 2{\leftrightarrow}3, \dots$) |
| `targeted` | 恒定目标类 $c^\*$ |
| `random` | 均匀采样自 $\{0,\dots,C-1\}\setminus\{y\}$ |

### 2.2 Per-round 随机化

`attack/hallucination.py: prepare_for_round`:每轮重采样

- 翻转比例 $\rho^t \sim \mathcal{U}[\rho_{min},\rho_{max}]$,区间由
  `hallu_flip_ratio_range` 指定;
- 翻转掩码与随机目标类,种子 $\text{seed}^t = s_a \cdot 100003 + t$($s_a$ 为攻击者专属种子,大素数避免多攻击者碰撞)。

效果:攻击梯度方向逐轮非平稳(CSE / local acc 震荡),比冻结的错误标签流形更接近现实攻击者。

### 2.3 隐蔽性(stealth)论证

由于攻击者跑的是**真实的本地 SGD**(只是标签被污染),更新的几何统计量(norm、cosine)自然落在 benign 分布附近,即约束

$$
\lVert \omega_a - \omega'_a \rVert \le \epsilon
$$

**天然满足**,不需要显式投影。`attack/hallucination.py: camouflage_update`
是恒等映射。这正是纯几何防御(Krum/norm-clipping 等)难以检测该攻击的原因,
也是同时引入行为信号(CSE 与 probe 行为视图)的动机。

**与经典 Byzantine baseline 的关键区别**:`sign_flipping`/`gaussian`/`alie` 是 dataset-free 的(不读本地数据、直接伪造更新);Hallucination 攻击者**使用**自己的本地数据。

---

## 3. 防御总览:三个 trust 模式

`hmp_gae/runtime.py: HMPGAERuntime.aggregate`

三个模式共享同一个检测统计量——聚合前的逐客户端 full-test CSE(§5.1),
区别只在"检测之后如何定权":

| trust_mode | 机制 | 角色 |
|---|---|---|
| `v4_cse_reject` | CSE flag → 常数乘子 | detect-then-suppress 消融臂 |
| `v5_cse_reject` | CSE flag → 线性 ramp 乘子 | V8 的 Stage A;matched-run 安全基线 |
| `v8_hmp_cse_propagation` | V5 种子 + 双视图超图传播 | 当前机制 |

V4/V5 是**无状态纯决策规则**:不构建超图、不训练 GAE、无跨轮状态。V8 每轮
在服务器端执行(全部在 CPU,因 $N$ 小):

```
Δ (N×d) ──固定 JL 投影──▶ u (N×64) ──mutual k-NN──▶ H^u(更新视图,当轮固定)
P (N×K×C) ──pairwise JS──▶ S^b ──mutual k-NN──▶ H^b(行为视图,当轮固定)
          ──双视图共识──▶ H^c(传播超图)
η 特征提取 ──HMP encoder(residual+LN,signed 输出)──▶ Z (N×32)
          ──cosine decoder──▶ Â;固定拓扑自监督训练若干 step (Adam)
CSE ──V5 决策层──▶ 高置信种子 F ──超图传播(§5.3)──▶ α ∈ Δ^{N-1}
          ──加权聚合──▶ Δ_g = Σᵢ αᵢ Δᵢ;──EMA 更新 Z_hist──
```

encoder/decoder 参数与 Adam 状态**跨轮持久化**(不重置),即 GAE 随 FL 轮次
持续在线学习;这是 V8 仅有的跨轮状态(连同 $Z^{hist}$)。

---

## 4. V8 的 GAE 组件

### 4.1 节点特征 $\eta_i$

`hmp_gae/node_features.py`

$$
\eta_i = f_{enc}\big(\underbrace{R\,\Delta_i}_{\text{JL 投影}},\ \underbrace{c_i}_{\text{上下文统计}},\ \underbrace{h_i^{t-1}}_{\text{历史嵌入}}\big) \in \mathbb{R}^{64}
$$

**(a) 固定随机投影(JL embedding)**(`node_features.py: compute_node_features`):
$R \in \mathbb{R}^{d \times 64}$,$R_{jk} \sim \mathcal{N}(0, 1/64)$,固定种子、**跨轮不变、不训练**——保证特征几何在时间上稳定(否则历史嵌入会活在漂移的基底里)。

**(b) 上下文统计** $c_i \in \mathbb{R}^4$(`node_features.py: context_stats`):

$$
c_i = \big[\ \log(1 + \lVert\Delta_i\rVert_2),\ \ \text{mean}(\Delta_i),\ \ \text{std}(\Delta_i),\ \ \cos(\Delta_i, \bar{\Delta})\ \big], \quad \bar{\Delta} = \tfrac{1}{N}\sum_j \Delta_j
$$

**(c) 历史嵌入** $h_i^{t-1} = z^{hist}_i \in \mathbb{R}^{32}$(EMA 缓存,见 §4.6;冷启动为零向量)。

**(d) $f_{enc}$**:2 层 MLP,$\text{Linear}(64{+}4{+}32 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 64)$,可训练。

### 4.2 双视图 k-NN 超图(当轮固定)

`hmp_gae/hypergraph.py: knn_hypergraph / semantic_js_similarity / knn_hypergraph_from_similarity`

每个节点 $i$ 生成一条以自身为中心的超边:

$$
\varepsilon_i = \{i\} \cup \text{top-}k\big(\ \operatorname{sim}(i, j),\ j \ne i\ \big), \qquad k_{eff} = \min(k, N{-}1)
$$

- **$M = N$**(每节点一条超边),使关联矩阵 $H \in \{0,1\}^{N \times N}$ 是方阵、decoder 维度跨轮稳定;
- $H_{i,e} = 1 \iff i \in \varepsilon_e$;度 $d_V(i) = \sum_e H_{i,e}$,$d_E(e) = k_{eff} + 1$;度逆以向量形式保存。

**更新视图**不使用可训练的 $\eta$ 构图,而使用固定 JL 投影:

$$
u_i=R\Delta_i,\qquad
\varepsilon_i^u=\{i\}\cup\operatorname{topk}_{j\ne i}\cos(u_i,u_j)
\;\Rightarrow\; H^u .
$$

**行为视图**只使用共享 probe 上的 per-sample softmax $P_i^{(q)}$,不使用
probe label(label 只可用于平衡采样,`semantic_probe_stratified`):

$$
\operatorname{JS}_{ij}=\frac1K\sum_q\frac12\left[
  \operatorname{KL}(P_i^{(q)}\Vert M_{ij}^{(q)})+
  \operatorname{KL}(P_j^{(q)}\Vert M_{ij}^{(q)})
\right],\quad
M_{ij}^{(q)}=\frac{P_i^{(q)}+P_j^{(q)}}2,
$$

$$
S^b_{ij}=1-\frac{\operatorname{JS}_{ij}}{\log 2},\qquad
\varepsilon_i^b=\{i\}\cup\operatorname{topk}_{j\ne i}S^b_{ij}
\;\Rightarrow\; H^b .
$$

令 $M^v_{ij}=1$ 当且仅当 $i,j$ 在视图 $v$ 中互相选择
(`hypergraph.py: mutual_neighbor_adjacency`),则共识关系

$$
C=M^u\odot M^b .
$$

传播超图 $H^c$ 以每个节点为中心,包含自身及 $C$ 中与其互邻的节点
(`hypergraph.py: consensus_propagation_hypergraph`)。因此单一更新几何偶合
或单一行为偶合都无权传播风险。两个视图与 $H^c$ 都在当轮构建一次并 detach,
**不随 GAE 训练 step 重建**。

### 4.3 HMP Encoder(两阶段超图消息传递)

`hmp_gae/encoder.py`

每层 $l$ 先 node→hyperedge,再 hyperedge→node,全部在固定的 $H^u$ 上;
V8 每层加入残差与 LayerNorm,以在极小的 $N=7$ 图上保留客户端特异内容:

$$
E^{(l)}=\operatorname{ReLU}\!\left((D_E^u)^{-1}(H^u)^\top
Z^{(l)}W_E^{(l)}\right),
$$

$$
Z^{(l+1)}=\operatorname{LN}\!\left((D_V^u)^{-1}H^uE^{(l)}W_V^{(l)}
+Z^{(l)}W_{skip}^{(l)}\right).
$$

- $Z^{(0)} = \eta$;dropout 施加在 $E^{(l)}$ 上;实现上 $W$ 先作用于特征再乘关联矩阵(数学等价);
- **最后一层不做 ReLU**,允许 signed latent(cosine decoder 需要负相似度);
- $L = 2$ 层,维度 $64 \to 64 \to 32$,输出 $Z \in \mathbb{R}^{N \times 32}$。

### 4.4 GAE Decoder

`hmp_gae/decoder.py`

**成对邻接(normalized cosine decoder)**,固定尺度 $\gamma = 4$:

$$
a_{ij}=\gamma\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2},
\qquad \hat A_{ij}=\sigma(a_{ij}).
$$

signed latent + cosine 归一使非邻居可以得到负 logit 和 $\hat A_{ij}<0.5$;
固定 $\gamma$ 避免在 $N=7$ 的在线问题上再学一个标定参数。

**超边关联(线性投影 decoder)**(`decoder.py: HyperedgeDecoder`):

$$
\hat{H}_{i,e} = \sigma(z_i^\top w^{dec}_e), \qquad \hat{H} = \sigma(Z\, W_{dec}^\top) \in [0,1]^{N \times M},\ M = N .
$$

$W_{dec} \in \mathbb{R}^{N \times 32}$ 可训练。BCE 用 logits 版本以保证数值稳定。

### 4.5 自监督损失(固定拓扑重构)

`hmp_gae/losses.py: total_loss_v8`

$$
\mathcal L_{V8}=\lambda_H\operatorname{BCE}_{w^+}(H^u,\hat H_{logits})+
\lambda_A\operatorname{BCE}_{i\ne j}(A^u,a)+
\lambda_{hist}\mathcal L_{hist}+\lambda_{wd}\lVert\theta\rVert_2^2 .
$$

- **(a) 超边重构 BCE**(带类不平衡加权,`losses.py: recon_loss_H`):
  $w^+ = \text{clamp}(\#\{H{=}0\}/\#\{H{=}1\},\ 1,\ 10)$;
- **(b) 固定拓扑邻接 BCE**(`losses.py: adjacency_recon_loss`):目标是
  detach 的 direct-mutual 更新邻接 $A^u = M^u$,而**不是** decoder 自己产生的
  平滑权重——这切断了旧路径"可学习 $\eta\to H\to$ 重构同一个 $H$"的自指反馈环。
  排除对角(每个节点与自身平凡相同,否则会主导 $N=7$ 的小损失);其每节点
  均值(`losses.py: per_node_adjacency_recon_error`)即 V8 的
  `v8_recon_error` 诊断;
- **(c) 历史一致性**(`losses.py: hist_loss`,$Z^{hist}$ detach,冷启动为 0):
  $\mathcal{L}_{hist} = \frac{1}{N d_z} \lVert Z - Z^{hist} \rVert_F^2$。

当轮所有优化 step(个数、学习率由 `defense_config` 决定)都使用同一个
$H^u$;训练完成后以 eval 模式重新前向一次,得到用于传播算子的 $\hat A$。

### 4.6 历史嵌入 EMA($Z^{hist}$)

`runtime.py: _update_history`

$$
z^{hist}_i \leftarrow \beta_h\, z^{hist}_i + (1 - \beta_h)\, z_i^t
$$

首次观测直接初始化 $z^{hist}_i = z_i^t$;$\beta_h$ = `hist_ema_beta`。
$Z^{hist}$ 进入:①节点特征的 $h_i^{t-1}$ 输入;②$\mathcal{L}_{hist}$。

---

## 5. CSE 决策层与 V8 传播

### 5.1 检测统计量:pool-median CSE 比值

`hmp_gae/trust_scorer.py`(三个模式共用)

服务器聚合前在 full-test 集上评估每个本地模型的 CSE(§7.2 定义),得

$$
r_i=\frac{\operatorname{CSE}_i}{\max(\operatorname{median}(\operatorname{CSE}),\ \epsilon)} .
$$

- 这是**绝对尺度**统计量,刻意不做 pool-relative z-score:相对打分没有绝对
  下限,在干净联邦里必然把最异质的 benign 当替罪羊;
- pool median 必须由 benign 多数控制:runtime 在构造时强制
  $0 < K_B < N/2$($K_B$ = `num_byzantine`,同时是 rank cap),否则规则反转;
- flag 集合(V4/V5 完全相同):

$$
F=\{\,\text{top-}K_B \text{ by } r\,\}\ \cap\ \{\,r_i>\tau\,\}, \qquad \tau = \texttt{v4\_tau\_ratio}.
$$

两个条件缺一不可:rank cap 独自承载归档重放的 zero-false-positive 性质
(无 cap 时 36 个 benign 误标,有 cap 时 0),比值下限使干净联邦不会永远
flag 自己的 top-$K_B$。$\tau = 1.85$ 为预注册常数(docs/DECISION.md "V4")。

### 5.2 V4 / V5 乘子

**V4**(`trust_scorer.py: v4_cse_reject_weights`):

$$
m_i = \begin{cases} \texttt{v4\_reject\_mult} & i \in F \\ 1 & \text{否则} \end{cases}
$$

`v4_reject_mult`$=0$ 是预注册的 hard-removal 消融臂(detect-then-remove)。

**V5**(`trust_scorer.py: v5_cse_reject_weights`),flag 集合与 V4 逐位相同,
乘子改为比值的线性 ramp:

$$
t_i = \operatorname{clip}\!\left(\frac{r_i - \tau}{r_{hard} - \tau},\ 0,\ 1\right), \qquad
m_i = \begin{cases} m_{floor} + (1 - m_{floor})(1 - t_i) & i \in F \\ 1 & \text{否则} \end{cases}
$$

刚过 $\tau$ 的边界 flag(benign 误标最可能的形态)几乎不损失权重;
$r \ge r_{hard}$ 的清晰证据饱和到 $m_{floor}$(此时与 V4 取
`reject_mult`$=m_{floor}$ 逐位相等)。$m_{floor} > 0$ 恒成立(硬置零是
FoolsGold 的机制,已否决)。

### 5.3 V8:CSE 种子驱动的双视图超图传播

`trust_scorer.py: v8_hmp_cse_propagation_weights`(Stage A 即
`v5_cse_reject_weights`,逐字节相同)

**(a) 传播算子。** 先由共识超图计算 node→edge→node 算子,删除对角并对
超过 1 的行归一(`hypergraph.py: hypergraph_propagation_matrix`):

$$
P=\operatorname{RowNorm}\!\left(\operatorname{OffDiag}\left[
(D_V^c)^{-1}H^c(D_E^c)^{-1}(H^c)^\top\right]\right),
$$

再用 GAE affinity 衰减,而**不把衰减后的质量重新归一到 1**:

$$
T=P\odot\hat A,\qquad \sum_jT_{ij}\le1 .
$$

这个"次随机"约束是必要的;否则节点只有一条边时,任意弱 affinity 都会被
重新放大成 1,GAE 实际不起作用。

**(b) 联合证据与共享 rank cap。** V5 的 flag 集合 $F$ 作为不可被替换的种子:

$$
q_i=\sum_{j\in F}T_{ij},\qquad
e_i=\operatorname{clip}\left(\frac{r_i-1}{\tau-1},0,1\right),\qquad
J_i=q_i e_i .
$$

只有 $i\notin F$、$J_i>0$ 且 CSE flag 后仍有 rank-cap 余额
($|F| < K_B$)的客户端,才能按 $(J_i,r_i,-i)$ 降序进入传播集合 $G$;
CSE flag 永远优先。传播客户端的连续乘子为

$$
m_i^{prop}=1-(1-m_{floor})J_i ,
$$

没有新增可调阈值:关系弱或 CSE 证据弱都只造成轻惩罚,两者同时强才逼近
$m_{floor}$。最终 $m_i$ 对 $F$ 取 V5 ramp、对 $G$ 取 $m_i^{prop}$、其余
精确取 1。

**结构保证与可证伪条件。** 无 V5 种子、无通向种子的双视图共识路径、无
$r_i>1$ 的同伴,或 rank cap 无余额时,代码直接返回 V5 tensor,逐元素相同。
超图是否产生增量必须从 `v8_propagated_flagged`、`v8_joint_evidence`、
`v8_consensus_edge_count` 与 `v8_propagation_matrix` 判断;若全程无传播,
实验结论就是 V8 退化为 V5,不能把 CSE 的收益归因于超图。反过来,若两个
攻击者都未形成 CSE 种子,V8 也不会自行检测;这是防止几何 scapegoat 的保守
边界。该模式仍依赖服务器 full-test local CSE,且与只伪造 update、不改变
`client.model` 的 `crafts_update` 攻击不兼容(server 与 defense facade 都
会显式报错)。

---

## 6. 聚合与退化

### 6.1 数据量加权聚合

$$
\alpha_i=\frac{D_i\, m_i}{\sum_j D_j\, m_j}, \qquad
\Delta_g = \sum_{i=1}^{N} \alpha_i\, \Delta_i, \qquad
w_{t+1} = w_t + \eta_{server}\, \Delta_g \quad (\eta_{server} = 1.0)
$$

(`trust_scorer.py: weighted_aggregate`;`server.py: run_round`)

- $D_i$ = 客户端数据量(benign 用真实 `len(data_indices)`,attacker 用
  `claimed_data_size`);
- **detection 与 weighting 解耦**:信任机制只负责压制 attacker
  ($m_i < 1$);benign 之间按自然数据量权重聚合,保留协作学习收益;
- 干净轮(无 flag、无传播)$m \equiv 1$,权重**逐位**等于 $n_k$ 先验——
  无替罪羊税,这是 V4 起的结构性质;
- `keep_min` 与 rank cap $< N/2$ 共同保证未 flag 的多数始终携带正质量。

### 6.2 退化与安全边界

- **$N \le 2$**:HMP 消息传递不适定,由
  `defense/__init__.py: HMPGAEDefense.aggregate` 自动回退 FedAvg;运行时异常也
  逐轮回退并记录 `fallback_reason`;
- **缺失输入即崩溃**:CSE 向量缺失/含 NaN,或 V8 缺 probe distributions,
  在 facade 与 runtime 双层显式 raise,绝不静默退化成 50 轮 FedAvg;
- **冷启动**:V4/V5/V8 的决策统计量(CSE)从 round 0 即生效,无 warmup;
- **V8 → V5 逐元素退化**(§5.3)是安全性质,不是性能证据。

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
- 与 accuracy/loss 共享同一次测试集前向(每轮免费);同时逐客户端计算 local acc / local CSE(本地模型 $w_t + \Delta_i$ 在服务器测试集上)——后者就是 §5 的检测统计量。

### 7.3 PPL — 困惑度(FL 结束后一次性)

`evaluation_hallucination.py` + `decoder_adapters.py`

把最终 LoRA 微调的 backbone 迁移进 `AutoModelForCausalLM`(SeqCLS → CausalLM,需 `save_global_checkpoint=True`),在类分层的测试子集上:

$$
\text{PPL} = \exp\Big( \frac{1}{n} \sum_{j=1}^{n} \text{NLL}_j \Big)
$$

$\text{NLL}_j$ = 第 $j$ 条样本的 shifted-label 平均 token 负对数似然(HF `outputs.loss`)。另报 per-class PPL。encoder-only backbone 优雅跳过。

---

## 8. 配置与复现边界

[`main.py`](../main.py) 的 `main()` 内 `config` 字典是唯一权威配置源。本文档定义
参数的数学含义,但不保存当前模型、数据集、轮数、阈值或实验名快照,因为这些
值会随实验臂变化。

复现某次运行应使用结果文件中归档的完整 config 和对应 commit。跨版本比较还需
遵守 [docs/DECISION.md](DECISION.md) 中预注册常数、matched-run 与
falsification 契约,不能从当前 `main()` 反推旧实验配置。复现已移除的
V1–V3/V6/V7 臂需 checkout 2026-08-11 之前的 commit。

---

## 9. 论文写作要点(叙事逻辑备忘)

1. **V8 的互补分工**是核心论点:CSE 提供高精度种子,update/probe 双视图超图保留关系并提升漏检同伴的 recall,GAE affinity 只做连续衰减;任何单通道都不能独立扩权。
2. **detection、传播与 weighting 解耦**:CSE 决定不可替换的 seed,HMP 只使用剩余 rank cap,最后仍回归数据量 FedAvg——避免权重集中破坏协作学习。
3. **绝对尺度 vs 相对打分**:检测统计量是绝对的 pool-median CSE 比值;pool-relative 打分(z-score 族)没有绝对下限,干净联邦必产生替罪羊——这是被否决的 V1–V3 路线留下的核心教训(历史见 DECISION)。
4. **消融链**:V4(检测 + 常数抑制)→ V5(+ 分级 ramp)→ V8(+ 超图传播)。归因超图必须用 matched V5/V8;V4/V8 对比混杂了 ramp 与超图两个因素。
5. **符号对照**:$\eta_i$ ↔ node_features.py;$H^u, H^b, H^c, T$ ↔ hypergraph.py;encoder 残差层 ↔ encoder.py;$\hat{A}, \hat{H}$ ↔ decoder.py;$\mathcal L_{V8}$ ↔ losses.py;$r, F, J, m, \alpha$ ↔ trust_scorer.py。
