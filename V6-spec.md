# HMP-GAE V6 修改规格（交付给 coding agent）

代码仓库：`/Users/lancecai/Documents/GitHub/HMP-GNN`（branch `main`，remote `github.com/GuangLun2000/HMP-GNN`）
本规格所有行号基于 commit `879fe26`（2026-08-05）。实现前请先 `git pull` 并核对行号。
`import torch` 在 Mac 上不可用，训练与 `tests/` 只能在 Colab 跑；本地只做静态修改与 review。

---

## 0. 先读：为什么不是"回退到 V3"

这一节是设计约束的来源，不是背景介绍。**实现前必须理解，否则会做出错误的取舍。**

### 0.1 "V3 accuracy 最高"不成立，且 accuracy 在本 benchmark 上不可用于排序

全库 69 个 run 实测（6 个 cell，canonical arm = seed 42 / N=7 / flip[0.3,0.8] / 50 轮）：

- Qwen AG non-IID：**完全无防御的 FedAvg-under-attack 排 mean acc 第 1（0.889493）**，V3 第 2（0.889467），差 0.000027 = 75,000 次预测里的 **2 次**。
- Llama Yahoo non-IID：**无防御 FedAvg 又是第 1**（0.675280），V3 第 2。
- Qwen AG non-IID 与 Llama AG non-IID 上，**无攻击的 clean ceiling 排在被攻击且无防御的 run 之下**（0.882973 vs 0.889493）。
- V3 在它存在的 4 个 cell 里**从未拿过第 1**；唯一一个"防御拿第 1"的 cell（Llama AG non-IID）拿第一的是 **V4**，不是 V3。
- 归档里现有 6 组 seed 复现对（seed 42 vs 42069/0，2026-08-03/04/05 加入）：seed-to-seed |Δmean acc| 中位数 **0.0207**。49 个相邻名次差里 **42 个（86%）小于最小实测 seed 噪声**。V3 与 V4 在 4 个 cell 的 accuracy 差分别是 median seed 噪声的 11.2% / 68.3% / −6.5% / 20.5%，**没有一个超过噪声带**，而且符号不一致（Llama AG 上 V4 更高）。

> 结论：accuracy 只能当作 utility 下限检查（本库只有 Qwen AG non-IID 的 Krum 0.7831 与 Qwen Yahoo non-IID 的 multi_krum 0.5413 真正跌破）。**不要把 accuracy 写进防御排序，也不要为 accuracy 做架构取舍。** 与 `docs/DECISION.md` 2026-07-30「The Attack Has No Accuracy Cost」一致，本次把该结论从 2 个 Llama cell 扩展到 6 个 cell 中的 4 个。

### 0.2 V3 的代价是真实且巨大的

同 cell 同 seed，V3 相对 V4（正数 = V3 更差）：

| cell | mean CSE | final CSE | PPL | trust separation |
|---|--:|--:|--:|---|
| Qwen AG non-IID | +37.8% | +53.7% | +9.7% | 2.10 → 16.13（V4 高 669%）|
| Qwen Yahoo non-IID | +73.0% | +190.6% | **−12.2%（V3 更好）** | **0.958** → 8.99 |
| Llama AG non-IID | +10.9% | −0.5% | +5.5% | 12.91 → 16.13 |
| Llama Yahoo non-IID | +63.3% | +76.5% | +31.4% | **0.994** → 8.99 |

**V3 在两个 Yahoo cell 上 trust separation < 1，即攻击者拿到的聚合权重比良性客户端平均值还高，连续 40 轮。** 这不是"稍弱"，是防御方向反了。

### 0.3 决定性实测：几何信号在 Qwen AG 上对确认攻击者是错的

我用归档 CSV 做了离线重放。对每一轮按 V4/V5 规则选出被 flag 的客户端（全部 5 个重放 run 里 **benign 误报 = 0**），再查同一轮几何管线算出但被丢弃的 `sigmoid_gate`：

| run | flagged 次数 | 攻击者上的几何 gate 均值 | gate<0.5（几何同意）比例 |
|---|--:|--:|--:|
| Qwen AG non-IID V4 | 97 | **0.766** | 16% |
| Qwen AG non-IID V5 | 97 | **0.649** | 32% |
| Llama AG non-IID V4 | 98 | 0.286 | 76% |
| Llama Yahoo non-IID V4 | 96 | 0.606 | 39% |

gate 越接近 1 = 几何越认为"该客户端是良性"。**在 Qwen AG 上，对已确认的攻击者，几何平均说"给它 77% 权重"。** 所以任何让几何去决定惩罚力度（即允许几何把惩罚调轻）的设计，都会在 Qwen AG 上把 CSE 打回 V3 水平。

### 0.4 决定性实测：V3 的 gate 架构本身有上限

我把绝对 CSE 作为第五个通道注入 V3 的 trust logit 后重放（`z_cse = log(r)/log(τ)`，融合进 `s`，再走 `sus = −s/weight_norm` → `sigmoid`），扫 `cse_weight`：

| cse_weight | Qwen AG separation | Llama AG | Llama Yahoo |
|--:|--:|--:|--:|
| 0（=V3） | 1.16x | 2.10x | 1.36x |
| 1.0 | 1.73x | 8.08x | 2.17x |
| **2.0（最优）** | **2.04x** | **8.49x** | **2.47x** |
| 4.0 | 1.59x | 1.72x | 1.58x |
| 8.0 | 1.07x | 1.07x | 1.07x |

separation 在 `cse_weight≈2` 见顶后**反而下降**。原因是结构性的：`weight_norm = sqrt(Σ weights²)` 随通道权重增长，`sus = −s/weight_norm` 把新增信号又除掉了；阈值 `reject_z_threshold=2.5` 是"每信号 robust-z 单位"，单一主导通道几乎不可能超过它。

> **即使把 CSE 最优地塞进 V3 的门控，Qwen AG 上 separation 上限约 2.0x，而 V4/V5 是 16.13x —— 差 8 倍。**
> 所以：**瓶颈不是信号，是 V3 的 gate/归一化架构。** 不要试图靠调 `graph_weight / semantic_weight / hist_weight_beta / cse_weight` 把 V3 救回来（而且 `semantic_weight=2.0` 与 `hist_weight_beta` 调参已被 `docs/DECISION.md` 明确否决）。

### 0.5 真正未解决的问题不是 accuracy，是 Yahoo 的 PPL

用户目标是"accuracy、CSE、PPL 同时最好"。按 cell 拆开看，这个目标**在两个 AG News cell 上 V4/V5 已经达成**：

- Qwen AG non-IID，V5：acc 0.8868（与第 1 在噪声内）、final CSE **0.0365**、PPL **69.71**、ppl_class_std **9.35** —— CSE / PPL / class_std **三项都优于无攻击 clean ceiling**（0.0409 / 71.40 / 14.95）。
- Llama AG non-IID，V4：acc **0.9139（cell 第 1）**、mean CSE **0.1067**、PPL **49.01**、class_std **9.68** —— 四项全优于 ceiling（0.9085 / 0.1077 / 56.88 / 11.19）。

失败只发生在 Yahoo 的 PPL：

- Qwen Yahoo non-IID V4：CSE 0.6378（优于 ceiling 0.6551）但 PPL **1209.10**，比 attack floor（1092.07）和 clean ceiling（1109.56）都差。
- Llama Yahoo non-IID V4：CSE 0.5095（优于 ceiling 0.5253）但 PPL 620.23 vs ceiling 431.07。

机制（`docs/DECISION.md` 2026-07-29 已记录）：10 类 Dirichlet-0.5 下把 7 个客户端里的 2 个压到 0.1×，**丢掉的是标签覆盖度，不是幻觉**。这是覆盖度问题，不是信任评分问题，**换 trust 架构解决不了**。见 §5。

---

## 1. 本次要做什么

拆成两个独立目标，**不要混在一次改动里**：

- **目标 A（本规格 §2–§4）**：论文一致性。让 HMP-GAE 的几何真正参与 α 的产生，且**在构造上不可能让 CSE 变差**。这是可以现在做、风险可证明有界的改动。
- **目标 B（本规格 §5）**：Yahoo PPL。需要另一套机制 + 一条新的 `docs/DECISION.md` 条目，**不在本次实现范围内**。

### 目标 A 的问题陈述

论文 Eq (21) + Algorithm 1 line 16–17 定义：`s_i = f_trust(z_i)`，`α_i = softmax(s_i)`，`Δθ_global = Σ α_i ω_i`。
V4/V5 实际执行的是 `α = normalize(m_i · n_i)`，`m_i` 只由 local-CSE 比值决定。整条超图/VGAE 管线（`graph_residual`、`recon_residual`、`sem_div`、`hist_dev`、`sigmoid_gate`、`sus_z`）**算了、记了日志、乘进了 0**。

实证佐证：V4/V5 的 trust separation 在**不同模型之间逐位相同** —— Qwen AG V4 = Llama AG V4 = Qwen AG V5 = **16.1289**（benign α 0.195160，attacker α 0.012100，三者完全一致，但 `trust_weights.csv` 的 md5 互不相同）；Qwen Yahoo V4 = Llama Yahoo V4 = **8.9906**。α 只是 (数据划分, seed) 的确定性函数，**不含任何模型信息，也不含任何几何信息**。

另外 `docs/DECISION.md`（archive，2026-07-29「Norm-Based Screening Is Ruled Out」）写着几何通道"remain the α-producing signal in V4; only the rejection decision moved to CSE" —— **这句话与代码不符**。本次改动正是让这句话变成真的。

---

## 2. V6 设计：`v6_cse_reject_geo`

### 2.1 机制

```
# ---- Stage 1：候选集。与 V4/V5 完全一致，不改。这一步承载 zero-FP 记录 ----
r_i        = cse_i / max(median_j(cse_j), eps)
max_flags  = min(max(0, k_cap), max(0, N - max(1, keep_min)))
flagged    = { i ∈ top-max_flags by r_i : r_i > tau }          # tau = v4_tau_ratio = 1.85（预注册）

# ---- Stage 2：CSE 证据强度。与 V5 的 ramp 完全一致，不改 ----
t_i        = clamp((r_i - tau) / (r_hard - tau), 0, 1)         # r_hard = v5_r_hard = 2.5（预注册）
m_cse_i    = m_floor + (1 - m_floor) * (1 - t_i)               # ∈ [m_floor, 1]

# ---- Stage 3：几何一致性。新增，这一步让 f_trust 真正参与 ----
# sus_i 已经在 runtime.py:490-497 每轮为所有 trust_mode 算好（含 EMA 平滑），直接用，无需新计算
g_i        = sigmoid(-soft_reject_k * (sus_i - reject_z_threshold))   # ∈ (0,1)，高 = 几何认为良性

m_i        = m_cse_i * (geo_floor + (1 - geo_floor) * g_i)     if i ∈ flagged
m_i        = 1.0                                                otherwise

alpha      = normalize(n_i * m_i)
```

### 2.2 为什么是这个形状（三条硬性质，实现时必须保住）

1. **单调安全性 —— CSE 不可能回退。** 因为 `geo_floor + (1-geo_floor)·g_i ∈ [geo_floor, 1]`，所以恒有 `m_i ≤ m_cse_i`。即 **V6 给任何被 flag 客户端的权重永远 ≤ V5 给的权重**。几何只能"加重"惩罚，永远不能"减轻"。结合 §0.3 的实测（Qwen AG 上几何对攻击者说 0.766），这是唯一不会把 CSE 打回 V3 的复合方式。
2. **完美退化 —— `geo_floor = 1.0` ⇒ 与 V5 逐位相同。** 这是回归守卫：第一次跑 V6 就用 `geo_floor=1.0`，输出必须与归档 V5 run 完全一致；不一致说明接线有 bug。
3. **clean-federation 恒等式保住（invariant 9）。** 未被 flag 的客户端 `m_i` 恒等于 `1.0`（**不是** `g_i`），所以无攻击时 `α = n_i/Σn` 精确成立，`tests/test_trust_robustness.py` 的 `test_no_attack_no_scapegoat` 仍然通过。**这一点是 V3 做不到的**：V3 的 gate 永远不等于 1，干净联邦里也会冤枉最异质的那个良性客户端。

### 2.3 预期效果（可证伪）

用归档 gate 值估算被 flag 攻击者的 `m`（`m_cse ≈ m_floor = 0.10` 在稳态）：

| cell | 几何 gate 均值 | `geo_floor=0.5` 下的 m | vs V5 的 0.10 |
|---|--:|--:|--:|
| Qwen AG non-IID | 0.766 | 0.0883 | −12% 攻击者权重 |
| Llama AG non-IID | 0.286 | 0.0643 | −36% |
| Llama Yahoo non-IID | 0.606 | 0.0803 | −20% |

即：几何**确实改变 α**（12–36%），方向恒为收紧，且量级可控。这既让 Eq (21) 在论文里成立，又不赌几何的可靠性。

### 2.4 论文对齐

- Algorithm 1 line 15–16 保持不变：`s_i = f_trust(z_i)` 仍然算，`g_i` 就是它经 sigmoid 读出的结果。
- 需要在 §IV 加**一句**方法描述：trust 读出与一个基于探针语义熵的绝对证据项做**保守合取**（conservative conjunction），并说明这样做的动机 —— 更新空间几何在有界偏差伪装下不可靠（本项目自己的实证：clean 与 attacked 联邦的 aggregate update norm 在 AG News 上 4 位小数完全相同，Llama Yahoo 上相差 1.53%）。
- **这句话是加分项不是补丁**：它把"update-space 几何单独不够、必须引入输出空间证据"变成论文的一个实证贡献，而不是一个方法与实现不符的漏洞。

### 2.5 明确不做（以及为什么）

- ❌ **不要**让几何减轻惩罚（`m_i > m_cse_i`）。§0.3 实测直接否决。
- ❌ **不要**把 `trust_mode` 换回 `soft_reject_fedavg` 再加通道。§0.4 实测：separation 上限 ~2.0x vs V4 的 16.13x。
- ❌ **不要**改 `v4_tau_ratio`(1.85) 或 `v5_r_hard`(2.5)。两者都已预注册（archive `docs/DECISION.md` 2026-07-29；repo `docs/DECISION.md`:87-94），看到结果后不得回调。
- ❌ **不要**把 `m_floor` / `reject_mult` 设为 0.0（repo `docs/DECISION.md`:95-99；硬清零是 FoolsGold 的机制，本库最差 PPL）。唯一预授权的 sweep 值是 {0.05, 0.02}。
- ❌ **不要**把 `local_cse` 送进 `_zscore`（invariant 7，`trust_scorer.py`:438-444）。
- ❌ **不要**顺手加 coverage-aware reweighting / per-class trust / sticky flags / cold-start holdback —— 这四项在 repo `docs/DECISION.md`:113-123 被显式推迟，需要新 DECISION 条目。

---

## 3. 代码改动清单（逐文件）

### 3.1 `hmp_gae/trust_scorer.py`

**新增函数**，插在 `v5_cse_reject_weights` 结束（:619）之后、`weighted_aggregate`（:622）之前：

```python
def v6_cse_reject_geo_weights(
    local_cse: torch.Tensor,      # (N,) 绝对 per-client CSE，禁止 z-score
    data_sizes: torch.Tensor,     # (N,) 原始 n_i，不封顶
    gate: torch.Tensor,           # (N,) 几何 sigmoid gate ∈ (0,1)，由 runtime 传入
    tau_ratio: float = 1.85,
    k_cap: int = 2,
    m_floor: float = 0.10,
    r_hard: float = 2.5,
    geo_floor: float = 0.5,
    keep_min: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
```

- Stage 1/2 直接复用 `v5_cse_reject_weights`(:505-619) 的实现，**逐行照抄，不要重写**（flag 用严格 `>`，`max_flags` 公式见 :591）。
- Stage 3 只在 `flagged` 位置乘 `(geo_floor + (1-geo_floor)*gate)`。
- 返回 diag：V5 的 5 个 key（`ratio` / `flagged` / `multiplier` / `ramp_t` / `median`）**再加** `geo_gate`(Tensor N)、`geo_mult`(Tensor N，即 `geo_floor+(1-geo_floor)*g`)、`m_cse`(Tensor N，Stage 2 的中间值，用于诊断几何贡献了多少)。
- 防御式校验：`0 < geo_floor <= 1`，`gate.numel() == N`，`gate` 落在 `[0,1]`（clamp 而非 raise，sigmoid 数值边界）。
- 更新模块头注释块（:33-46，现有 V4/V5 段落）。

### 3.2 `hmp_gae/runtime.py`

1. **import**：:35-43 加 `v6_cse_reject_geo_weights`（挨着 :41/:42）。
2. **`_known_modes`**：:160-163 加 `"v6_cse_reject_geo"`。**漏掉这一步会在 `__init__` 直接 ValueError**（:164-168）。
3. **旋钮解析**：仿 :192-193，加 `self.v6_geo_floor = float(self.cfg.get("v6_geo_floor", 0.5))`。放在同一块，保证对其它 mode 惰性。
4. **校验**：:194 的 `if self.trust_mode in ("v4_cse_reject","v5_cse_reject")` 加入新 mode，以复用 `2*num_byzantine < N`(:199-204) 与 `tau > 1.0`(:205-208) 两条守卫；再仿 :215-226 加 `0.0 < v6_geo_floor <= 1.0`（**注意上界是闭区间**，因为 1.0 是 V5 等价点，必须合法）与 `r_hard > tau`。
   > **所有校验必须放在 `__init__`。** 从 `runtime.aggregate` 内抛的 ValueError 会被 `defense/__init__.py`:236-248 吞掉并静默降级成 FedAvg 跑满 50 轮。
5. **dispatch**：:521 的分支条件加入新 mode（或另起 `elif`），函数体仿 :528-562。几何输入**已在作用域内**，无需新算：`trust`(:467-483)、`sus_raw`(:490-496)、`sus_used`(:497)。gate 的算法见 `gate_diagnostics`(:684-719)，公式 `sigmoid(-soft_reject_k*(sus-reject_z_threshold))`(:718)——**必须用 EMA 后的 `sus_used`，不是 `sus_raw`**，与 V3 语义一致。`used_mode`(:562) 设为 `"v6_cse_reject_geo"`。
6. **stats**：:676-691 的 `if v4_diag is not None` 块扩展。现有代码是 `if trust_mode == "v4_cse_reject": ... else: <V5 keys>` —— **这个 else 会把 V6 错当成 V5**，必须改成三分支。新增 `v6_geo_floor` / `v6_geo_gate` / `v6_geo_mult` / `v6_m_cse`。
7. **checkpoint**：V6 与 V4/V5 一样无状态（:721-725），**不要**新增跨轮状态。若将来加 sticky flag，必须同步 :740-748 / :750-759，否则 resume 会静默重置。

### 3.3 `defense/__init__.py`

:216-217 的 `_tm in ("v4_cse_reject","v5_cse_reject")` 加入新 mode。这一检查刻意位于 FedAvg try/except（:227）**之前**，作用是让 plumbing bug 大声崩溃而不是静默降级。`local_cse` 已在签名(:197)里且无条件转发(:234)，**无需改签名**。

### 3.4 `server.py`

1. :114-117 `_needs_local_cse` 的 tuple 加入新 mode。**必须**，否则 `local_cse` 恒为 `None`。此改动同时打开"每轮聚合前 local eval"（:965-983 + :1006-1010 复用短路），会覆盖 `eval_local_every_n_rounds`。
2. :433-457 控制台诊断：现有分支用 `'v5_m_floor' in defense_stats` 区分 V5/V4，**V6 会落进 V4 的措辞**。加第三个 case，打印 `geo_floor` 与该轮 `geo_mult`。
3. :489-502 的 defense_stats key 白名单加入 §3.2.6 的所有新 key。**不在这个白名单里的诊断永远进不了 result.md**。
4. :99-103 `_needs_probe` 只看 `semantic_weight > 0`，V6 保持 `semantic_weight=1.0` 即自动开启，无需改。

### 3.5 `main.py`

- :1584 `'trust_mode'` 改为 `'v6_cse_reject_geo'`；同步更新 :1551-1583 的说明块。
- 在 :1618-1619（V5 旋钮）旁加 `'v6_geo_floor': 0.5`，并写明它是**预注册值**。
- 更新 :1221 起的 run 说明块：写清这是 V6 首测、对照的归档 companion、以及"`geo_floor=1.0` 必须复现 V5"这条 sanity check。
- `defense_config` 支持一层深合并（:1749-1758），所以 sweep 可以只翻 `trust_mode` / `v6_geo_floor`。

### 3.6 `check_docs.py`

- `SYMBOLS["hmp_gae/trust_scorer.py"]`(:51-55) 加 `v6_cse_reject_geo_weights`。
- 顺手补 `v5_cse_reject_weights` —— **当前它不在列表里，V5 入口点没有被守卫保护**（现存缺陷）。
- 该守卫只校验 `def/class <NAME>` 存在，不校验签名或行为。

### 3.7 `tests/test_trust_robustness.py`

必须新增（无法在 Mac 跑，Colab 执行）：

1. **等价性测试（最重要）**：`geo_floor=1.0` 时 `v6_cse_reject_geo_weights` 的输出与 `v5_cse_reject_weights` **逐元素相等**（`torch.allclose`，`atol=0`）。仿 :474-487 的饱和等价测试写法。
2. **单调安全性**：任意 `gate ∈ [0,1]`、任意 `geo_floor ∈ (0,1]`，恒有 `alpha_v6[flagged] <= alpha_v5[flagged]`（归一化前比 `m`，避免归一化混淆）。
3. **clean-federation 恒等式**：无 flag 时 `alpha == n_i/Σn` 精确成立（仿 :418 / :462）。
4. **单调性**：更高的 CSE ratio 不得拿到更轻的 multiplier（仿 :451）。
5. **runtime 端到端**：仿 :594-650 的 V5 用例。
6. 在 `__main__` 列表（:686-692）注册。

### 3.8 归档侧 `tools/extract_csvs.py`（在 OneDrive 归档里，不在代码仓库）

路径：`.../results-for-Hallucination/tools/extract_csvs.py`，`SUBHEADING_MAP`（:37-58）。
只有当 Colab notebook **打印**了新表格时才需要加映射，否则数据只存在于 `result.md`。建议一并补上现存缺口：`v4_ratio` / `v4_flagged` 至今没有 CSV schema（archive `AGENTS.md` 的 "Known gap"）。

---

## 4. 必须遵守的不变量

代码强制（违反会崩）：

1. `trust_mode` 必须在 `_known_modes` 内 —— `runtime.py`:160-168。
2. `0 < num_byzantine` 且 `2*num_byzantine < num_clients` —— :199-204。rank cap 必须 < N/2，否则 CSE 池中位数不再由良性客户端主导。
3. `v4_tau_ratio > 1.0` —— :205-208。
4. `0.0 < m_floor < 1.0` —— :215-220；`v5_r_hard > v4_tau_ratio` —— :221-226。
5. `local_cse` 长度必须为 N —— `runtime.py`:528-540 / `defense/__init__.py`:212-222 / `server.py`:975-981。
6. `local_cse` 必须是绝对全测试集统计量，**绝不 z-score** —— `trust_scorer.py`:438-444。
7. `data_sizes` 保持原始、不封顶 —— `trust_scorer.py`:456-458。
8. 权重非负且和为 1，并保留退化兜底 —— :489-494 / :605-610 / :845-850。
9. `max_flags = min(max(0,k_cap), max(0, N - max(1,keep_min)))`，`keep_min` 个客户端永不可能全被 flag —— :478 / :591。
10. flag 需要**两个条件同时成立**（rank cap **且** `r > tau`）。**承载 zero-FP 记录的是 rank cap，不是 tau** —— :428-436。
11. `N <= 2` 一律回落 FedAvg —— `defense/__init__.py`:204-210。
12. CSE-reject 系模式与伪造更新型攻击者（`crafts_update` / AugMP）不兼容 —— `server.py`:966-973 会 raise。V6 继承此限制。

政策强制（不会崩，但是硬约束）：见 §2.5。

### 已知的两处文档/代码不一致（实现时会撞上）

- **A**：archive `docs/DECISION.md`（2026-07-29）称几何通道在 V4 里"remain the α-producing signal" —— 与代码不符（V4/V5 里 `α = normalize(m_i·n_i)`）。**V6 落地后应更新这条**。
- **B**：**"V5" 这个名字被重载了**。archive `docs/DECISION.md`（2026-08-05，arm-B 预注册）里的"two-sided V5 rule"（用于抓 arm B 那种低熵、自信而错的攻击者）**与已实现的单边 graded ramp `v5_cse_reject` 不是同一个东西**，且前者尚未实现。命名时不要混淆，V6 也不要顺手把它实现进去。

---

## 5. 目标 B：Yahoo PPL（**本次不实现**，需先立 DECISION 条目）

问题：Qwen Yahoo non-IID V4 的 CSE 0.6378 优于 clean ceiling 0.6551，但 PPL 1209.10 比 attack floor 1092.07 和 clean ceiling 1109.56 都差；Llama Yahoo V4 PPL 620.23 vs ceiling 431.07。

机制：10 类 Dirichlet-0.5 下把 2/7 客户端压到 0.1×，丢的是**标签覆盖度**。这是覆盖度问题，换 trust 架构（含本规格的 V6）解决不了 —— V6 只会让惩罚更重，PPL 只会更差。

候选方向（按代价排序，**都需要新 DECISION 条目**）：

1. **提高 `m_floor`**（最便宜）：直接在 CSE↔PPL 之间移动工作点。现有预授权 sweep 值 {0.05, 0.02} 是**更狠**的方向；更软的值（0.2 / 0.3）需要新条目。可先在 Qwen Yahoo 上做单轴扫描。
2. **Coverage-aware 再分配**：把被 flag 客户端的权重按类覆盖重新分给良性客户端。repo `docs/DECISION.md`:113-123 显式推迟。且服务端不知道客户端的私有标签分布，需要新的估计机制（可从 probe 上的 per-class 预测分布近似）。
3. **保留更新方向、只投影掉幻觉分量**：最有原理但代价最大，本库无先例。

**报告纪律**：任何关于 Yahoo CSE 收益的表述都必须同时给出 PPL 回归（archive `docs/DECISION.md` 2026-07-29「Y18 CSE Gain Must Be Reported With Its PPL Regression」）。

---

## 6. 另外两个便宜且高价值的修复（建议同批做）

### 6.1 把 `probe_cse` 落到归档（几乎零成本）

`hmp_gae/runtime.py`:701-702 **已经**在算 per-client 的探针语义熵：

```python
probe_cse = -(Pp * Pp.log()).sum(dim=-1).mean(dim=1)
stats["probe_cse"] = probe_cse.detach().cpu().tolist()
```

它也已在 `server.py`:497 的白名单里。但它**从未出现在任何归档 `result.md`** —— 因为 Colab notebook 没有打印它。

只要在 notebook 里加一个打印小节 + 在归档 `tools/extract_csvs.py` 的 `SUBHEADING_MAP` 加一行，下一次 run 就能拿到 `probe_cse.csv`。价值：这是**唯一能验证 §6.2 那个泄漏修复是否可行的数据**，而且完全免费。

### 6.2 测试集泄漏（论文层面的真实风险，建议至少先记录）

`server.py`:513-559 的 `evaluate_local_metrics` 遍历的是 **`self.test_loader`** —— 即**服务器测试集**。而对外报告的 global CSE 也来自同一个测试集。

也就是说：**V4/V5 的拒绝信号，就是被当作头号结果报告的那个指标，在同一份数据上算出来的。** docstring 自己也承认了（"Using the server's public test set is inherent to FedLLMs evaluation."）。审稿人几乎必然会问："你的防御按你自己报告的指标来筛客户端。"

补充：探针也取自 `test_loader.dataset`（`server.py`:694 / :636），所以 V2/V3 的 `sem_div` 同样是测试集派生 —— 只是它是 label-free 且 pool-relative，不是被报告的那个量，对抗性弱一些。

修复路径（**建议单独一次改动，不要和 V6 混**）：
1. 先做 §6.1，拿到 `probe_cse`，离线比较 `probe_cse` 比值与 `local_cse` 比值的判别力（100 个分层样本 vs 1500 个全测试样本；熵是样本均值，标准误按 1/√n 缩放，而攻击者/良性比值在 4–20x 量级，**大概率仍可分**，但必须实测确认）。
2. 若判别力足够，把 V6 的 `cse_i` 输入从全测试 `local_cse` 换成 `probe_cse`。
3. 更彻底的做法：让探针改从一个独立的 held-out split 抽样，而不是 `test_loader.dataset`。这才能完全关掉这个口子。

---

## 7. 实验计划与预注册成功判据

**顺序执行，每一步是下一步的前提。**

- **Run 0（sanity，必须先跑）**：`trust_mode='v6_cse_reject_geo'`，`v6_geo_floor=1.0`，其余与归档 V5 companion（`20260805-...-v5-...`，Qwen AG non-IID seed 42）完全一致。
  **通过判据**：`trust_weights.csv` 与该 V5 run **逐位相同**。不同 = 接线有 bug，停下来修，不要继续。
- **Run 1（主测）**：同配置，`v6_geo_floor=0.5`。对照 = 同一个 V5 run。
- **Run 2**：Llama AG non-IID，`geo_floor=0.5`，对照 `20260730-agnews-llama-HMPGAE-v4版本-汉霖✅`。选这个 cell 是因为几何在这里最可靠（gate 均值 0.286，同意率 76%），V6 的收益应当最大。
- **Run 3（只在 Run 1/2 都通过后）**：Llama Yahoo non-IID，对照 `20260728-yahoo-llama-HMPGAE-v4-汉霖`。

**预注册成功判据（跑之前写死，看到结果后不得改）**：

V6 相对同 cell 的 V5/V4 对照，**四项同时不回归**：
1. `mean CSE` 与 `final CSE` —— 不得变差。§2.2 的单调安全性保证了这一点在构造上成立，若观察到回归，说明实现有 bug。
2. `PPL` —— 相对回归不超过 +2%（V5 vs V4 的实测差是 +0.3%，作为噪声尺度参考）。
3. `ppl_class_std` —— 不得变差。
4. `mean accuracy` —— 只做下限检查，**不得**用来排序（§0.1）。判据：不低于该 cell 的 clean ceiling 减去 2pp。

**额外必须报告的量**：`geo_mult` 的每轮分布。若它在所有轮次都 ≈1.0，说明几何实际上没起作用，V6 相对 V5 只是换了个名字 —— 这本身就是必须如实报告的负面结果，**不要靠调 `geo_floor` 把它调出效果来**（那就变成看到结果后调参了）。

**单一 seed 的纪律**：归档现有 6 组 seed 复现对显示 mean acc 的 seed 噪声中位数 0.0207、mean CSE 13.10%。任何小于该带宽的差异都必须报成"打平"，不得称为改进。

> 顺带：archive `AGENTS.md` 的 "Metric Conventions" 仍写着 "All archived runs are seed 42; there is no seed variance in the archive" —— **这句已过期**（2026-08-03/04/05 已加入 6 组 seed 复现），应当更新。
