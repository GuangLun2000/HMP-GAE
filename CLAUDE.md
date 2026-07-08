# CLAUDE.md — HMP-GNN

> 详细功能/配置/命令见 [README.md](README.md)。本文件只补充无法从代码直接推导出来的上下文与约定。

## What this is

- **课题**：Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing（工作中，未投稿）
- **核心思路**：客户端发起 label-flip hallucination 攻击；服务器端用 HMP-GAE 给每个客户端打信任分（α），按 α 鲁棒聚合
- **三信号防御**（trust_scorer 的 `s = -(graph + recon + semantic + hist)`）：
  1. **graph_residual**：k-NN 超图孤立度（attacker 在超图中触达的 benign 节点少 → 高 residual）
  2. **recon_residual**：GAE 解码出的 A_hat 上的孤立度（encoder 训练后变锐利）
  3. **semantic_div**：固定 probe set 上 per-sample softmax 的 pairwise symmetric KL（行为指纹，几何 stealth 攻击也藏不住）
  4. hist_dev：与 EMA 历史 latent 的距离（默认权重 0，因为 benign drift 反而比 attacker 大）

## Workflow: 本地不训练（critical）

Mac 仅做代码编辑；训练在 Google Colab A100。**含义对 Claude 至关重要**：

- 不要建议 "先在本地跑一下 `python main.py` 验证"——Mac 单轮都跑不动
- 出 bug 优先靠静态分析 / trace 推理，不要要求 "先跑一下看看"
- macOS 上 `torch.cuda.is_available()` 永远 False——涉及 device 的代码要兼容 CPU 路径
- 用户自己 commit；改完代码不要主动 git add/commit
- **唯一 Colab notebook 是 [HMP_GAE_Colab.ipynb](HMP_GAE_Colab.ipynb)**。不要复制/新建 `*_Colab*.ipynb` 变体——Colab 同时只能跑一个，维护多份只会漂移；跑不同配置走 `COLAB_CONFIG_OVERRIDES` 或修改 main.py 的 config

## Canonical config: [main.py](main.py) 的 `main()` config dict

[main.py:885](main.py#L885) 是**唯一**权威 config 源。任何参数调整都在这里改：

- 改默认行为 → 改 main() 的 config
- 对照实验 → 在 Colab Step 3 用 `COLAB_CONFIG_OVERRIDES` 临时覆盖（跑完即恢复），或调用 [main.py:1141](main.py#L1141) 的 `run_suite()`
- 不要在 notebook cell 里硬编码超参覆盖（除非临时尝试，跑完即撤）

**当前默认实验** = `hmpgae_hallu_randflip_n7_r50_qwen`：N=7 (5 benign + 2 attackers)，50 轮，Qwen2.5-0.5B + LoRA(r=8)，AG News (IID, 10K subset)，per-round 随机化 label-flip（ratio∈[0.3, 0.8]），hmp_gae 三信号防御。

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
| [evaluation_hallucination.py](evaluation_hallucination.py) | 全局 PPL 评估（FL 结束后一次；encoder-only 优雅跳过） |
| [decoder_adapters.py](decoder_adapters.py) | SeqCLS → CausalLM backbone 迁移（PPL 评估前置） |

新算法沿用上表里的符号命名；新符号在对应文件顶部 docstring 简注即可。

## Pitfalls（代码看不出但容易踩）

- **`results/` 是 gitignore**：所有 json/csv/png/checkpoint 都写到这里；不要 commit，不要让脚本默认写到别处
- **Attacker 的数据语义因 attack 类型而异**：
  - `Hallucination`：**使用**自己的本地数据但训练时翻转 label（dataset-USED with flips）
  - `sign_flipping` / `gaussian` / `alie`：**dataset-free**，不读自己的数据，只伪造 update
  - main.py:112-114 那条 "attackers do NOT use these local data" 注释**只对后三种成立**——读到这条不要"修正"代码
- **N ≤ 4 时 HMP-GAE 自动 fallback 到 FedAvg**（[defense/\_\_init\_\_.py](defense/__init__.py)）：超图信号在小 N 下不稳；动这里要同步更新 README 的 limitations
- **`defense_config.device: 'cpu'` 故意的**：HMP-GAE 子模型很小（N=7），CPU 比反复 GPU↔CPU 搬数据快
- **`semantic_weight > 0` 会触发 server 每轮 per-client probe forward**：从 test_loader 头部确定性取 `semantic_probe_size` 条样本；`semantic_weight=0` 时整条 probe path 跳过
- **`gate_signal` 在 `semantic_weight > 0` 时自动从 `'graph'` 升级为 `'combined'`**（[hmp_gae/runtime.py:110-114](hmp_gae/runtime.py#L110-L114)），除非显式在 config 设；改信号融合权重时小心这条隐式 promotion
- **`hallu_flip_map` 的 key 可能是 str**（从 JSON config 读时会变成字符串），[main.py:316](main.py#L316) 有 normalize 逻辑，不要破坏
- **requirements.txt 受 Colab 镜像约束**：特别是 `torchao` / `peft` / `transformers` 版本兼容（见文件内注释 line 11-13）
- **LoRA `target_modules=None` 含义是"用 PEFT 默认"**：默认对 DistilBERT 友好；换冷门 backbone 时可能需要显式列出 attn projection 名
- **CSE 每轮免费**（共享 test forward），**PPL 是 FL 结束后一次性**（需要 checkpoint）——不要把 PPL 计算挪进每轮循环

## Out of Scope（V1 不主动做，除非明说）

- 真正的多模态 encoder（V1 用 LoRA-only 模拟多模态）
- `dirichlet_alpha < 0.3` 的极端 non-IID 调优
- 改 byzantine baseline (`sign_flipping`/`gaussian`/`alie`) 的对外行为
- 复制/新建 `*_Colab*.ipynb` 变体

**已在路线图上**（不算 OOS，但需明示再动）：Krum / Median / FLTrust baseline 防御；stealth attacker（cosine + norm projection 隐蔽变种）；多 seed 报告 mean±std。
