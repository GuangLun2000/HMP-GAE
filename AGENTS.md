# AGENTS.md — HMP-GNN

> 本文件只维护代码无法直接表达的协作约束和高风险陷阱。运行方式见
> [README.md](README.md)，文档职责见 [docs/README.md](docs/README.md)，当前公式见
> [MATH_LOGIC.md](MATH_LOGIC.md)，版本历史与实验契约见
> [docs/DECISION.md](docs/DECISION.md)。

## 项目上下文

- 课题：Hallucination Immunization for Multimodal Federated LLMs via
  Hypergraph Message Passing（研究原型，未投稿）。
- 攻击：Hallucination 客户端使用自己的本地数据，并在训练时翻转标签。
- 当前主路径：V8 以 V5 的 full-test CSE 判定为高置信种子，利用 update/probe
  双视图一致超图传播风险，再按数据量加权聚合。
- 机制边界：超图没有独立 flag 权；无种子、无双视图路径或无剩余 rank cap
  时，V8 必须逐元素退化为 V5。完整公式只在 `MATH_LOGIC.md` 维护。

## 不可违反的工作流

- Mac 只做编辑和秒级静态检查；完整 FL 训练在 Google Colab GPU 上进行。
  不要要求本地运行 `python main.py` 来“试一下”。
- 唯一 notebook 是 [HMP_GAE_Colab.ipynb](HMP_GAE_Colab.ipynb)。不要复制或新建
  `*_Colab*.ipynb`；notebook 只调用裸 `main()`，不覆盖配置。
- 用户自行提交代码；不要主动 `git add`、commit 或 push。
- 保留工作区内与当前任务无关的未提交改动，不要回滚或覆盖。
- macOS 上 `torch.cuda.is_available()` 为 False；新增 device 路径必须兼容 CPU。

## 验证矩阵（不训练）

| 改动范围 | 必跑检查 | 说明 |
|---|---|---|
| 任意 `.md` | `python check_docs.py` | 链接、配置键、符号和 anti-drift 检查 |
| 任意 `.py` | `python -m compileall -q .` | 只解析语法，不需要数据集或 GPU |
| `hmp_gae/` 或 trust/runtime 路径 | `python tests/test_trust_robustness.py` | CPU 回归；需要 PyTorch |

测试全绿只证明静态契约和局部不变量，不能证明 FL 收敛或性能提升。

## 配置唯一来源

[`main.py`](main.py) 的 `main()` 内 `config` 字典是唯一权威配置。没有 CLI、
notebook 或环境变量覆盖入口。

- 默认行为和所有 A/B 实验臂都只改该字典。
- 每个实验臂必须使用唯一的 `experiment_name` 和 checkpoint 子目录；不要让
  resume 误接另一条实验轨迹。
- 不在文档中复制易变的模型、数据集、轮数、阈值或实验名；需要当前值时读
  `config`，归档结果时保存完整 config。
- 不引用 `main.py` 行号；只引用 `main()` 或配置键名。
- 预注册常数、禁止事后调整的规则和历史版本差异只在
  [docs/DECISION.md](docs/DECISION.md) 维护。

## 模块职责

| 文件 | 修改边界 |
|---|---|
| [`main.py`](main.py) | 配置、数据分区、客户端装配、实验入口 |
| [`server.py`](server.py) | 轮次编排、聚合前 local CSE、probe forward、评估 |
| [`client.py`](client.py) | `BenignClient` 与 FedProx 本地训练 |
| [`attack/hallucination.py`](attack/hallucination.py) | label-flip 攻击与逐轮随机化 |
| [`defense/__init__.py`](defense/__init__.py) | 防御 facade、输入守卫与小 N fallback |
| [`hmp_gae/runtime.py`](hmp_gae/runtime.py) | HMP-GAE 端到端执行和跨轮状态 |
| [`hmp_gae/trust_scorer.py`](hmp_gae/trust_scorer.py) | 信号融合、V4–V8 决策、最终权重 |
| [`hmp_gae/`](hmp_gae/) 其余模块 | 节点特征、超图、encoder、decoder、loss |
| [`fed_resume.py`](fed_resume.py) | 逐轮断点和轨迹指纹 |
| [`evaluation_hallucination.py`](evaluation_hallucination.py) | FL 结束后一次性 PPL |
| [`tests/test_trust_robustness.py`](tests/test_trust_robustness.py) | trust 路径回归与安全退化不变量 |

## 高风险陷阱

- `results/` 被 gitignore；结果、图和 checkpoint 都应留在该目录。
- Hallucination attacker 会本地训练；SignFlipping、Gaussian、ALIE 是
  dataset-free 的伪造 update。不要混写两类语义。
- `num_clients <= 2` 时 HMP-GAE 才硬回退到 FedAvg；更大的小联邦只是信号可能
  偏弱，不等于触发 fallback。
- `defense_config.device='cpu'` 是有意设计：N 很小时，小型 HMP 子模型在 CPU
  上避免频繁 GPU/CPU 搬运。
- `semantic_weight > 0` 才触发逐客户端 probe forward；CSE-reject/V8 还要求
  聚合前的 `local_cse`。缺少 V8 所需的 probe distributions 必须显式报错。
- CSE-reject 使用服务器 full-test local eval，并强制逐轮计算。这是当前研究
  假设，也与“本地模型仍 benign、只伪造 update”的攻击不兼容。
- V8 是否实际使用超图只看 `v8_propagated_flagged`、`v8_joint_evidence`、
  `v8_consensus_edge_count` 等诊断；全零必须报告为 V8 退化到 V5。
- `hallu_flip_map` 从 JSON 读取后 key 可能变为字符串；入口负责转回整数。
- LoRA `target_modules=None` 表示使用 PEFT 默认值；更换 backbone 时核对目标层。
- PPL 只在 FL 结束后运行，且需要 checkpoint；不要移入逐轮循环。
- requirements 受 Colab 镜像中的 `torchao`、PEFT、Transformers 兼容性约束。

## 默认不扩展的范围

- 真正的多模态 encoder。
- 极端 non-IID 专项调参。
- 改变经典 Byzantine baseline 的对外行为。
- 新建第二份 Colab notebook 或第二套配置入口。

路线图中的多 seed、stealth attacker 和更多 baseline 只有在用户明确要求时再动。
