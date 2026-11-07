> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

> Historical Phase 1/2 record. Current Phase 3 scope and blockers: [release readiness](release-readiness.md), [AIDE decision](aide-release-decision.md), [public support](public-host-support.md). Earlier A3 scope restrictions are superseded only for the conditional external inference contract; generic AIDE LR feature fusion remains blocked.

# PPA 整理交付记录

第二阶段的 A–G 审计结论见 [Phase2 报告](phase2-release-audit.md)。本页保留第一阶段交付记录；Host 状态与阻塞项以第二阶段报告为准。

这是一个可运行、经过验证的**发布候选版本**。尚未满足完整论文公开复现和
正式开源发布的条件；具体缺口见 [Release blockers](release-blockers.md)。

## A. Repository Audit

见 [逐文件功能、调用链、正式结果关系与迁移建议](repository-audit.md)。
原仓库不改动；只在独立候选版本合并相关实现。历史实验、权重、数据、缓存、
第三方目录和未提交文件均保留在研究归档中。

## B. Final Repository Tree

```text
PPA/
├── README.md
├── LICENSE                         # 待所有者确定项目许可证
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-hosts.txt
├── ppa/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── response.py
│   ├── fusion.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   └── hosts.py
├── scripts/
│   ├── motivation.py
│   ├── ppr_only.py
│   ├── main_results.py
│   └── extract.py
├── configs/
│   ├── ppr.yaml
│   └── hosts.yaml
├── tests/test_ppa.py
├── examples/data_manifest.example.csv
└── docs/
    ├── delivery.md
    ├── repository-audit.md
    ├── cache-format.md
    ├── hosts.md
    ├── licenses.md
    ├── verification.md
    ├── release-blockers.md
    └── evidence/                   # 脱敏回放证据，不含数据/权重/特征缓存
```

## C. Core Implementation Summary

Motivation 保留 Q90 基础视图、三种扰动、五层原始响应的等权 global scalar，
输出逐图响应、域/标签、ROC-AUC、元数据和 layer-wise 可视化。

PPR-only 使用 L4/L5 的 Mean/人口标准差/Max 组成 1536D，源 TRAIN 拟合
StandardScaler 与 balanced LR，源 VAL 选 C。Feature Fusion 拼接 Host
embedding 与 PPR；Score Fusion 按 Host 保留 OOF-LR、原始加法或源 TRAIN
标准化加法。Effort 保留源 VAL BA 阈值。没有训练 Host 或添加新 MLP。

Evaluation 输出逐生成器、等权 macro 和 pooled 的 AP/ACC/ROC-AUC/BA；
单列 GenImage source-family 排除范围与 source-content-disjoint 范围，
记录阈值来源、split、标签方向及缓存/模型哈希。所有静态历史表都明确标注来源。

## D. Reproduction Commands

从候选仓库根目录运行，先按 README 安装核心依赖及官方 AlexNet 权重。
这些命令形式已用真实本地 manifest/cache 执行：

```bash
# A：单一冻结独立测试划分；jpeg 也可换成 blur / noise
python scripts/motivation.py --dataset data/motivation.csv --perturbation jpeg --output-dir outputs/motivation-jpeg --visualize 1

# 准备 B 所需的 JPEG PPR 缓存；manifest 同时含源 TRAIN/VAL 和目标 TEST
python scripts/extract.py --dataset data/experiment.csv --output cache/ppr.npz

# B：PPR-only，不需 Host
python scripts/ppr_only.py --dataset data/experiment.csv --cache cache/ppr.npz --output-dir outputs/ppr-only

# C：需原 Host 的冻结特征、原生 logit 和 PPR，以及完整来源元数据
python scripts/main_results.py --dataset data/experiment.csv --cache cache/clip-ppr.npz --host CLIP --fusion feature --output-dir outputs/clip-feature
```

Score route 使用 `--fusion score` 和新的 output-dir。其他 Host 的维度、
特殊协议、官方来源、版本与未验证范围见 [Host 说明](hosts.md)。模板不含真实
图像和可伪装成功的假哈希；精确 split/权重尚待发布，不能直接用模板复现论文。

## E. Verification Report

见 [完整 Passed / Failed / Not Tested / Missing Dependencies](verification.md)。
25 项测试全部通过；三种 Motivation 历史 AUC 回放完全一致；PPR-only 两个
完整目标集缓存回放一致；26 组 Host 路线/数据集抽样预测回放通过。真实图片
最小测试和独立导入通过。不要将这些结论扩大为所有 Host 原图端到端复现。

## F. Git Report

候选版本的 HEAD 指向 `release/ppa-open-source`，是没有 commit 的独立 Git
仓库，配置的 origin 是用户提供的 `https://github.com/kkgan406/PPA`。
未进行 commit、push、分支删除、历史改写或 GitHub Release。

候选文件全部为新建、未暂存、未提交；没有已跟踪文件的覆盖修改。新增文件见
上方目录树和 `git ls-files --others --exclude-standard`。本地数据/缓存/运行
结果不在候选目录中。原仓库仍在原分支和 HEAD，未提交变更原样保留。

建议后续 commit message（本次不自动提交）：
`refactor: prepare minimal PPA reproduction release candidate`

研究归档的 186 个可达历史 commit 已审计，存在绝对路径与实验资产，不应直接
推送该历史作为新发布仓库。候选历史尚为空；原历史未被破坏性清理。

## G. Release Blockers

见 [具体阻塞及原因](release-blockers.md)：AIDE 历史 MLP feature route 与
当前限制冲突、精确 manifest/Host checkpoint 获取方式、第三方授权、DINOv2/
CLIMP 原始代码与权重绑定、旧 GenImage 扩展域描述子数值来源、许可证及论文
正式信息。所有未知项保留 TODO 或明确报错，没有用编造代码/链接/结果补齐。
