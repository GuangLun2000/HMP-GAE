# AGENTS.md — HMP-GNN

> 详细功能/配置/命令见 [README.md](README.md)；算法符号与数学推导见 [MATH_LOGIC.md](MATH_LOGIC.md)。本文件只补充无法从代码直接推导出来的上下文与约定。

## What this is

- **课题**：Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing（工作中，未投稿）
- **核心思路**：客户端发起 label-flip hallucination 攻击；服务器端用 HMP-GAE 给每个客户端打信任分（α），按 α 鲁棒聚合
- **三信号防御**（trust_scorer 的 `s = -(graph + recon + semantic + hist)`）：
  1. **graph_residual**：k-NN 超图孤立度（attacker 在超图中触达的 benign 节点少 → 高 residual）
  2. **recon_residual**：GAE 解码出的 A_hat 上的孤立度（encoder 训练后变锐利）
  3. **semantic_div**：固定 probe set 上 per-sample softmax 的 pairwise symmetric KL（行为指纹，几何 stealth 攻击也藏不住）
  4. hist_dev：与 EMA 历史 latent 的距离（默认权重 0，因为 benign drift 反而比 attacker 大）

## Workflow: 本地不训练（critical）

Mac 仅做代码编辑；训练在 Google Colab A100。**含义对 Codex 至关重要**：

- 不要建议 "先在本地跑一下 `python main.py` 验证"——Mac 单轮都跑不动
- 出 bug 优先靠静态分析 / trace 推理，不要要求 "先跑一下看看"
- macOS 上 `torch.cuda.is_available()` 永远 False——涉及 device 的代码要兼容 CPU 路径
- 用户自己 commit；改完代码不要主动 git add/commit
- **唯一 Colab notebook 是 [HMP_GAE_Colab.ipynb](HMP_GAE_Colab.ipynb)**。不要复制/新建 `*_Colab*.ipynb` 变体——Colab 同时只能跑一个，维护多份只会漂移；跑不同配置走 `COLAB_CONFIG_OVERRIDES` 或修改 main.py 的 config

## Verify 一个改动（不训练）

跑不了 FL，但改完有几关可过（秒级、无需 dataset/GPU）：

- `python check_docs.py` —— 文档↔代码一致性守卫（stdlib，~0.1s）。**改任何 `.md` 后必跑**：校验 agent 文档里没有 `main.py:<行号>`、markdown 链接不死、docs 提到的 config key/符号在代码里真存在——是本文件 anti-drift 约定的可执行版本。**纯编辑 Mac 也能跑**。
- `python -m compileall -q .` —— 全库语法面（只解析不 import，无需 torch）。**改任何 `.py` 后跑**。**纯编辑 Mac 也能跑**。
- `python tests/test_trust_robustness.py` —— trust-scoring CPU 回归（含 legacy 路径 bit-for-bit），~1s。**碰 `hmp_gae/`（trust_scorer / runtime / hypergraph）后跑**。⚠️ **需要装了 torch 的环境**——纯编辑 Mac 通常没有 torch，这一关交给 CI 或 Colab。

三关全绿 ≠ 训练正确——收敛/精度验证仍在 Colab。

## Canonical config: [main.py](main.py) 的 `main()` config dict

`main()` 里的 `config` dict 是**唯一**权威 config 源。**不标行号**——main.py 频繁改动，任何行号都会失效；一律用符号定位（`main()` / `run_suite()` / config key 名）。任何参数调整都在这里改：

- 改默认行为 → 改 `main()` 的 `config`
- 对照实验 → 在 Colab Step 3 用 `COLAB_CONFIG_OVERRIDES` 临时覆盖（跑完即恢复），或调用 `run_suite()`
- 不要在 notebook cell 里硬编码超参覆盖（除非临时尝试，跑完即撤）

**不变的实验骨架**（很少动，可依赖）：N=7（5 benign + 2 attackers），50 轮，LoRA(r=8)，per-round 随机化 label-flip（`hallu_flip_ratio_range`），`attack_method='Hallucination'` + `defense_method='hmp_gae'` 三信号防御。

> **易变 knob（dataset / model_name / flip ratio 等）一律以 `config` dict 现值为准，本文件不复述具体值**——写死的数值随时会过时；需要当前值时读 `config`，不要引用本文件。（撰写本节时现值，仅供参考：Yahoo Answers non-IID(0.5) + Llama-3.2-1B。）

## 模块速查（哪里改什么）

| 文件 | 关心什么 |
|---|---|
| [hmp_gae/runtime.py](hmp_gae/runtime.py) | HMP-GAE 端到端运行入口；`Z_hist` (EMA latent) 在此 |
| [hmp_gae/trust_scorer.py](hmp_gae/trust_scorer.py) | 三信号融合 + soft/hard reject gating；`α_i` / `s` (combined logit) 在此 |
| [hmp_gae/{node_features,hypergraph,encoder,decoder,losses}.py](hmp_gae/) | 论文算法符号一一对应（`η_i` / `H, D_V, D_E` / `A_hat, H_hat`），公式即文件名 |
| [server.py](server.py) | 聚合编排、CSE 评估、per-round probe forward |
| [client.py](client.py) | `BenignClient` (FedProx) 基类 |
| [attack/hallucination.py](attack/hallucination.py) | 本论文攻击；per-round 随机化 flip 在此实现 |
| [attack/{sign_flipping,gaussian,alie}.py](attack/) | 经典 Byzantine baseline——**不要改对外行为**（V2 横向对比锚点） |
| [attack/augmp.py](attack/augmp.py) | AugMP 攻击（learned VGAE+GSP stealth model-manipulation，~3900 行，从 IoA-Attack-GRMP 移植）；懒加载，仅 `attack_method='AugMP'` 时 import，当前默认实验**不用** |
| [evaluation_hallucination.py](evaluation_hallucination.py) | 全局 PPL 评估（FL 结束后一次；encoder-only 优雅跳过） |
| [decoder_adapters.py](decoder_adapters.py) | SeqCLS → CausalLM backbone 迁移（PPL 评估前置） |
| [models.py](models.py) | `NewsClassifierModel`（SeqCLS + LoRA 装配；`lora_target_modules` 在此） |
| [data_loader.py](data_loader.py) | `DataManager` / 4 数据集加载 / tokenizer（**IID/non-IID 分区实际在 [main.py](main.py)**） |
| [fed_resume.py](fed_resume.py) | per-round 断点续跑（Colab 掉线恢复）；fingerprint 校验在此 |
| [tests/test_trust_robustness.py](tests/test_trust_robustness.py) | trust-scoring CPU 回归（legacy 路径 bit-for-bit）；改 trust_scorer/runtime 后跑 |

> **AugMP 隔离约定（token 节流）**：`attack/augmp.py` 约 3900 行、约 40K token，且当前实验不跑 AugMP（懒加载——`attack_method` 非 `'AugMP'` 时根本不 import，对运行时零成本）。**除非任务明确要改 AugMP，否则不要读取该文件**；全库阅读 / code review 时跳过它。其内部结构（VGAE 代理模型 + 增广拉格朗日约束优化：distance + 双边 cosine 约束）需要时再按需读取。

新算法沿用上表里的符号命名；新符号在对应文件顶部 docstring 简注即可。

## Pitfalls（代码看不出但容易踩）

- **`results/` 是 gitignore**：所有 json/csv/png/checkpoint 都写到这里；不要 commit，不要让脚本默认写到别处
- **Attacker 的数据语义因 attack 类型而异**：
  - `Hallucination`：**使用**自己的本地数据但训练时翻转 label（dataset-USED with flips）
  - `sign_flipping` / `gaussian` / `alie`：**dataset-free**，不读自己的数据，只伪造 update
  - main.py 分区打印处那条 `"attackers do NOT perform local training and do NOT use these local data"` 注释**只对后三种成立**——读到这条不要"修正"代码
- **N ≤ 2 时 HMP-GAE 自动 fallback 到 FedAvg**（[defense/\_\_init\_\_.py](defense/__init__.py) 的 `HMPGAEDefense.aggregate`，原因写入 `fallback_reason`）：这是代码里**唯一**的硬阈值——早期文档/README 写 N≤4 不准确（小 N 下超图信号确实偏弱，但真正触发回退是 N≤2）；动这里要同步更新 README limitations
- **`defense_config.device: 'cpu'` 故意的**：HMP-GAE 子模型很小（N=7），CPU 比反复 GPU↔CPU 搬数据快
- **`semantic_weight > 0` 会触发 server 每轮 per-client probe forward**：从 test_loader 头部确定性取 `semantic_probe_size` 条样本；`semantic_weight=0` 时整条 probe path 跳过
- **`gate_signal` 在 `semantic_weight > 0` 时自动从 `'graph'` 升级为 `'combined'`**（[hmp_gae/runtime.py](hmp_gae/runtime.py) 的 `HMPGAERuntime.__init__` 里 gate_signal auto-promotion 段），除非显式在 config 设；改信号融合权重时小心这条隐式 promotion
- **`hallu_flip_map` 的 key 可能是 str**（从 JSON config 读时会变成字符串），[main.py](main.py) 的 Hallucination 分支有 `{int(k): int(v) ...}` normalize，不要破坏
- **requirements.txt 受 Colab 镜像约束**：特别是 `torchao` / `peft` / `transformers` 版本兼容（见文件内注释 line 11-13）
- **LoRA `target_modules=None` 含义是"用 PEFT 默认"**：默认对 DistilBERT 友好；换冷门 backbone 时可能需要显式列出 attn projection 名
- **CSE 每轮免费**（共享 test forward），**PPL 是 FL 结束后一次性**（需要 checkpoint）——不要把 PPL 计算挪进每轮循环
- **`trust_mode: 'v4_cse_reject'`（V4，2026-07-28）改变 server 的评估时序**：server 会在**聚合前**对每个 client 做 full-test local eval（`Server._needs_local_cse`），并强制每轮 local eval（覆盖 `eval_local_every_n_rounds`）；缺 `local_cse` 会 loud crash（defense 层在 FedAvg-fallback try 之前校验，不会静默回退）。rank cap **复用 `num_byzantine`**（要求 < N/2，runtime 构造时校验）；`v4_tau_ratio=1.85` 是预注册值，跑完实验**不许回调**。与 AugMP（crafts_update）不兼容（local CSE 看不到伪造 update）。V4 信号**绝不过 `_zscore`**——被否决的替代方案与理由记录在 [docs/DECISION.md](docs/DECISION.md)
- **`trust_mode: 'v5_cse_reject'`（V5，2026-08-06）= V4 的 flag 判定 + 连续惩罚 ramp**：flag 规则与 V4 逐字节相同，只把被 flag 客户端的乘子从常数 `v4_reject_mult` 换成 CSE ratio 的线性 ramp（`trust_scorer.v5_cse_reject_weights`；τ 附近 ≈1.0，`v5_r_hard` 以上饱和到 `v5_m_floor`）。评估时序 / local_cse 硬要求 / AugMP 不兼容 / rank cap 全部继承 V4。`v5_r_hard=2.5` 是预注册值（由归档 V4 run 的稳态 attacker ratio 最小值标定），`v5_m_floor` 禁 0（硬归零已被否决）——设计决策与标定依据见 [docs/DECISION.md](docs/DECISION.md)。V5 尚未跑过确证实验；诊断 CSV 复用 `v4_*` 通道族

## Out of Scope（V1 不主动做，除非明说）

- 真正的多模态 encoder（V1 用 LoRA-only 模拟多模态）
- `dirichlet_alpha < 0.3` 的极端 non-IID 调优
- 改 byzantine baseline (`sign_flipping`/`gaussian`/`alie`) 的对外行为
- 复制/新建 `*_Colab*.ipynb` 变体

**已在路线图上**（不算 OOS，但需明示再动）：Krum / Median / FLTrust baseline 防御；stealth attacker（cosine + norm projection 隐蔽变种）；多 seed 报告 mean±std。
