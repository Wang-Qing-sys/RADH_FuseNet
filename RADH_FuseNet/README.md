# RADH-FuseNet：PyTorch 独立参考实现

> **这是依据用户提供的稿件编写的参考实现，不是作者官方源码，也不是已复现论文分数的代码。**
> 稿件链接的 GitHub 仓库在本次核查时仅有 README。原始 300 名驾驶员数据、完整标签协议和部分超参数未随稿件提供。演示数据只能验证程序运行，不能当作论文实验结果。

对应稿件：*Multi-level Reliability Aware Fusion with Dynamic Hypergraph Modeling and Wearable Sensing for Long-Term Driver Affective State Analysis*。

## 1. 安装

推荐 Python 3.11（支持 3.10 及以上）。这不是先前的 TensorFlow/Python 3.7 环境，请单独创建环境。

Windows / Anaconda 示例：

```bash
conda create -n radh python=3.11 -y
conda activate radh
cd RADH_FuseNet
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

有 NVIDIA GPU 时，请按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择与你的驱动兼容的版本，再安装其余依赖。本次核验使用 CPU，不能声称已在你的 GPU 上测试。

## 2. 一键跑通

在包的根目录执行：

```bash
python -m radh demo --output runs/demo --device cpu
```

此命令使用**明确标记的合成演示数据**和较小的模型配置，运行自监督预训练、监督微调、独立受试者测试和预测导出。它是软件冒烟测试，不是 300 名驾驶员实验，不是论文准确率验证。完整结构配置在 `configs/paper.json`，演示配置在 `configs/smoke.json`。

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

具体通过情况及环境见 `QA_REPORT.md` 和 `docs/VALIDATION.md`。未完成或未测试的项目会明确列出。

## 3. 用自己的数据训练、测试、预测

```bash
python -m radh train --data data/windows.npz --config configs/paper.json --output runs/experiment
python -m radh evaluate --data data/windows.npz --checkpoint runs/experiment/best.pt --output runs/experiment/test_metrics.json --robustness
python -m radh predict --data data/windows.npz --checkpoint runs/experiment/best.pt --output runs/experiment/predictions.csv
```

推荐提供自己的受试者划分：训练命令增加 `--split-json splits.json`，JSON 包含 `train`、`val`、`test` 三个受试者 ID 列表。**同一驾驶员的所有窗口必须在同一个集合。** 程序拒绝受试者交叉；归一化仅在训练集拟合，保存在检查点中供验证和测试使用。不要通过测试集选择超参数。

### NPZ 数据约定

| 键 | 形状与含义 |
|---|---|
| `x` | float32 `[N, 5, T]`，通道顺序 PPG、PPS、RSP、EHS、RHS |
| `subject_ids` | `[N]` 整数或字符串，不能使用 object 数组 |
| `mask` | 可选 bool `[N,5,T]`；True 表示有效观测，缺失处 False |
| `y` | 可选 int `[N]`，0–4 分别为 activated、deactivated、unpleasant、pleasant、neutral；-1 表示无标签 |
| `target` | 可选 float32 `[N,H]`，有真实连续强度标签才提供，缺失用 NaN |
| `timestamp` | 可选 `[N]`，窗口时间，用于追踪；不同驾驶员不能混用连续状态 |

这是本实现的显式输入格式，不是稿件公开的数据文件格式。**不能把离散分类标签当作连续强度标签。** 原始传感器单位、采样率、事件对齐和标签对应关系应由数据提供者确认。

CSV 转换入口：

```bash
python scripts/prepare_csv.py --input session1.csv session2.csv --output data/windows.npz --source-hz 100 --target-hz 32 --window 1024 --stride 512 --channels PPG PPS RSP EHS RHS
```

每个 CSV 只放一名受试者的一次会话，含 `subject_id` 和上述通道列；标签列的具体定义见脚本 `--help`。样例中的采样率、窗口长度和步长是**示例参数，不是论文确定值**。预处理不会把长缺失段编造成有效观测。

## 4. 模型组成

- 五个独立模态编码器：局部卷积、STFT 复数频域处理、选择性状态空间分支。
- 湿度补偿、信号质量描述、共享/私有特征与重建。
- 可靠性控制的动态超图：生理先验、邻近 patch 自适应关系、可靠性调制的消息传递。
- 1/2/4 三尺度状态空间时序融合。
- 三个生理模态分类头和一个全局分类头；EHS、RHS **没有独立情感分类投票权**。
- 可靠性折扣的 Dirichlet 意见融合、冲突处理、NIG 连续预测。
- 掩码预训练、干扰增强、监督优化、受试者独立评估和检查点预测。

源码模块尽量按功能拆分，使用标准 PyTorch，无需安装 Mamba CUDA 核或图神经网络扩展。选择性扫描是可读的原生 PyTorch 实现，**不是官方 Mamba 内核，也不承诺相同速度**。整体模型针对已完成的分析窗口；其中状态空间路径使用因果更新，但窗口质量统计、频谱分析等不应被误称为完整的逐采样流式因果系统。

## 5. 论文一致性与边界

详细映射、公式冲突以及采用的实现选择见 `docs/METHOD_AND_ASSUMPTIONS.md`。

主要已知限制：

1. 论文未交代最终采样率、窗口长度、预测跨度、训练轮数、batch size、全部损失系数及全部最终超参数。代码中的相应设置是可修改的参考值。
2. 部分公式、框图和文字不一致。例如补偿后的呼吸来源、三尺度门控的输入数量、证据折扣写法和冲突转移函数。实现优先采用可归一、数值稳定且符合文字物理角色的解释，并逐项标注。
3. 数据集 A 未提供。代码包没有其数据或预训练权重，没有复现论文表格分数。
4. WESAD 的生理通道和标签不等同于稿件的五分类/双湿度设置。使用它必须显式关闭湿度补偿、重新设置模态角色和类别；不伪造 EHS/RHS，也不把三类原标签随意映射成五种情感。
5. 所有附带的运行指标若来源于演示数据，只能说明软件流程可执行，不说明真实驾驶员情感识别能力。
6. 全缺失或只有环境上下文时，分类意见应退化到无知/均匀分布。正常输入的预测不确定性也未经过真实数据临床或安全认证。

本包仅用于科研开发，不是医疗诊断设备或可直接部署的驾驶安全决策系统。只使用经过授权的数据；不上传私人生理数据至外部服务。检查点和任何第三方 pickle 文件仅应从可信来源加载。

## 6. 可选预处理和外部数据配置

CSV 的 --wavelet 选项额外需要安装 PyWavelets；默认不启用。configs/wesad.json 是显式关闭湿度补偿的三分类适配示例，不包含 WESAD 下载、转换或完整 LOSO 实验。模型检查点保存训练状态，但当前 CLI 未提供断点续训命令。

附带 examples/smoke_output 是实际合成演示输出，不能用作论文成绩。
