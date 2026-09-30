# π0.5 LIBERO 微调与文本生成异常：训练代码及参数对照

在本次文本探针中，LIBERO checkpoint 的语言 Transformer 权重是不可读输出的直接原因：恢复这些层的 base 权重后，两个提示都生成可读文本；把这些层的 LIBERO 权重放进 base 后，两个提示都生成异常字符。单独恢复视觉编码器或词嵌入与输出归一化层，都没有恢复可读输出。

公开微调路径允许动作损失更新 VLM（视觉语言模型），却没有文本预测损失。参数对照与梯度结构共同支持“动作微调改写了中间语言表示，使文本读出失配”的解释。这些证据不能把原因单独归于 LIBERO 数据，也不能证明模型的全部语言理解能力消失；实际训练目标的因果作用还需要重新训练对照。

## 调查范围与证据

本次调查比较发布的 JAX `pi05_base` 与 `pi05_libero`，检查公开的 LIBERO 微调代码，并在同一 LIBERO 首帧上替换模型内存中的参数。本实验分别检查 VLM 的视觉编码器、语言 Transformer 层、词嵌入和输出归一化层。参数替换只用于定位文本输出异常，不执行机器人动作，不保存混合 checkpoint。

输入帧为已提交的 `docs/experiments/pi05_first_frame_compare.npz`，SHA-256 为 `2527af8d5fa09247dd84d8b198054623fe9b907ac70eb249d29f742ec1eb6813`。原始任务是：`pick up the black bowl between the plate and the ramekin and place it on the plate`。

此前，同一 JAX 解码代码下，base 能生成可读的图像描述、问答和子任务文本；LIBERO checkpoint 在三种提示下均生成异常字符。随后，对 Hugging Face `lerobot/pi05_libero_base` 的独立 PyTorch 检查也复现了异常，并核对了完整模型参数表、文件哈希及选定权重。Hugging Face 仓库名中的 `base` 不代表它包含未经 LIBERO 微调的 VLM。详见[此前的 LeRobot 检查](pi05_lerobot_libero_base_text_probe_zh.md)。

## 同一帧上的六组参数对照

实验使用 JAX CPU、BF16（bfloat16 精度）权重，每一步选概率最高的 token（词元），最多生成 8 个 token。所有组使用同一帧、同一图像预处理、同一官方 tokenizer 和同一解码代码，分别测试 `caption en` 与 `Task: {task}\nSubtask:`。base 的 CPU 输出与此前 GPU 运行的前 8 个图像描述 token、完整 5 个子任务 token 一致。LIBERO 的后续异常 token 在 CPU 与 GPU 间有数值差异；下表全部来自本次 CPU 运行，可以在同一执行环境内比较。

语言层替换覆盖 VLM 的 18 层 Transformer，包括注意力、前馈网络及层内归一化，明确排除动作专家的 `_1` 分支。词嵌入和最终归一化作为另一组替换；因为输入与输出共享词嵌入矩阵，这组替换也会改变文本输入嵌入。文本探针调用 `llm([embedded, None])`，动作专家不进入文本计算。每次调用 JAX 编译的函数时显式传入当前参数，避免复用最初被闭包捕获的权重。

下表的“高 ID 概率质量”是首个输出位置上，token ID 大于等于 `250000` 的归一化概率之和。这是明确的数值区间，不是通用的文本质量分类器；可读性依据实际文本与原始 token 序列判断。

| 模型与替换操作 | 图像描述提示的输出 | 子任务提示的输出 | 子任务首 token 的高 ID 概率质量 |
|---|---|---|---:|
| base，原始权重 | `The image shows a collection of objects arranged` | `pick up small plate` | 1.485% |
| LIBERO，原始权重 | 异常字符，首 token 为 `255700` | 异常字符，首 token 为 `255689` | 99.971% |
| LIBERO，只恢复 base 视觉编码器 | 异常字符，首 token 为 `255259` | 异常字符，首 token 为 `255629` | 97.127% |
| LIBERO，只恢复 base 词嵌入与最终归一化 | 与 LIBERO 基线的 8 个 token 完全相同 | 异常字符，首 token 为 `255689` | 99.965% |
| LIBERO，只恢复 base 语言 Transformer 层 | `Subtask: pick up bottle of` | `Subtask: pick up bowl` | 0.00128% |
| base，只替换为 LIBERO 语言 Transformer 层 | 异常字符，首 token 为 `255259` | 异常字符，首 token 为 `255693` | 96.893% |

正向恢复与反向替换提供了比权重差异更直接的证据：保持其他部分固定，仅改变语言层就能改变是否生成可读文本。因此，语言层变化足以在 base 的输入环境中重现异常，也能解释 LIBERO 模型在当前探针中的异常。视觉权重仍然影响输出分布；“单独恢复视觉编码器不足以修复”不等于视觉变化完全无关。

恢复语言层没有恢复完整的任务能力：图像描述提示被回答成了子任务格式，base 的 `pick up small plate` 也选错了原任务对象。本实验只有一张图像、两种提示和短文本，证实的是当前探针中的可读性变化，尚未检验一般语言基准、完整场景理解或 LIBERO 动作成功率。

[原始结果与各层权重差异](https://github.com/zhoujunlingla/openpi/blob/debug/pi05-text/docs/experiments/pi05_text_weight_ablation.json)包含全部 12 次输出、原始 token ID、首 token 前 10 个候选、概率分布指标、运行时间和各层参数变化；[复现实验脚本](https://github.com/zhoujunlingla/openpi/blob/debug/pi05-text/scripts/probe_pi05_text_weight_ablation.py)只在内存中替换参数。

## 公开的微调路径如何处理语言

### 数据中的任务文本进入了动作计算

LIBERO 转换脚本保存图像、腕部图像、状态、动作，并将原始 `language_instruction` 保存为 LeRobot 的任务文本。训练时，`PromptFromLeRobotTask` 根据任务索引取出文本，`TokenizePrompt` 将文本编码成输入 token；VLM 的 `embed_prefix` 把这些 token 与图像 token 一起送入 Transformer。因此，语言指令参与自动计算。

这条数据路径没有把图像描述或子任务文本保存为预测目标。文本生成探针显示的字符串是推理阶段临时解码出来的结果，并不是训练数据中的文本标签。相关代码：[数据转换](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/examples/libero/convert_libero_data_to_lerobot.py#L43)、[任务文本与 token 变换](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/transforms.py#L248)、[VLM 输入](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/models/pi0.py#L105)。

### VLM 可以更新，损失只约束动作

公开配置 `pi05_libero` 从 `pi05_base` 加载权重，使用 `physical-intelligence/libero`，`prompt_from_task=True`，训练 30,000 步，batch size 为 256，峰值学习率为 `5e-5`。该配置没有覆盖默认的 `freeze_filter=nnx.Nothing`，所以训练参数包含 VLM、视觉编码器和动作专家。

`Pi0.compute_loss` 计算连续动作的 flow matching 损失：先将噪声与真实动作插值，再让模型预测速度 `v_t`，目标为 `noise - actions`。返回值是两者的均方误差。代码取出 VLM 与动作专家两路输出，但只有动作专家的 `suffix_out` 进入 `action_out_proj` 和损失；VLM 最终的 `prefix_out` 没有进入文本损失。这条路径没有词表交叉熵、子任务文本损失或保持原模型文本分布的约束，也没有阻断动作损失向 VLM 传播的梯度。相关代码：[配置](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/training/config.py#L743)、[参数冻结默认值](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/training/config.py#L493)、[动作损失](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/models/pi0.py#L189)、[梯度计算](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/scripts/train.py#L145)。

这些配置描述的是公开、可复现的微调路径。发布 checkpoint 没有提供完整的训练日志与逐步权重，所以不能断言它的全部训练过程与这份配置逐项相同。

### 梯度可以改变中间表示，而不维护词表匹配

每层注意力先分别计算 VLM 与动作专家的 Q、K、V；Q 是查询，K/V 是供查询读取的键和值。动作专家用自己的 Q 读取包含 VLM 的 K/V，所以动作损失能改变 VLM 的 K/V 以及产生这些 K/V 的前层表示。

最后一层 VLM 的 Q、注意力输出投影、MLP（前馈网络）和最终归一化只影响最后的 VLM 输出；该输出没有进入动作损失。在当前计算图中，这些参数没有来自动作目标的梯度。最后一层 VLM 的 K/V 与注意力前归一化仍影响动作专家，因而能得到动作梯度。没有动作梯度并不意味着 checkpoint 中的数值必须完全相同，优化器、参数滑动平均和舍入也可能产生少量差异。参见[注意力实现](https://github.com/zhoujunlingla/openpi/blob/a9846efb8cba555a224aff3db943a7427e9d3ce5/src/openpi/models/gemma.py#L170)。

发布权重的差异与这个结构一致：此前完整测量的第 0 层 Q 相对均方根（RMS）变化为 19.83%，第 17 层 Q 为 0.007%；词嵌入中观测到的异常 token `255684` 对应行完全未变，最终归一化约 99.90% 的 BF16 值相同。本次各层采样又发现，第 17 层 K/V 的相对 RMS 变化约 16.27%，该层前馈归一化完全相同。相对 RMS 变化定义为 `RMS(微调权重 - base 权重) / RMS(base 权重)`，不等同于能力下降比例。

文本探针使用原模型共享的词嵌入矩阵作为词表投影，计算式为 `token 得分 = 最终隐藏表示 × 词嵌入矩阵的转置`。即使某个 token 的词嵌入未变，隐藏表示变化也能改变该 token 的得分。

因此，一个有证据支持的机制是：中间语言表示为动作目标发生变化，而用于文本生成的后续变换和词表投影缺少同步约束。这个机制能产生不可读文本；仅凭权重差异还不能证明模型内部的全部语言理解能力消失。

## 官方 Knowledge Insulation 方法提供的参照

Physical Intelligence 的 [Knowledge Insulation 说明](https://www.pi.website/research/knowledge_insulation)讨论了连续动作训练可能扰动 VLM 表示的问题。该方法让 VLM 用离散动作和一般视觉语言数据进行 token 预测训练，同时阻断连续动作专家传回 VLM 的梯度。推理阶段仍由动作专家生成连续动作。

当前公开的 `pi05_libero` 训练路径没有实现上述 token 预测训练与梯度阻断。官方方法支持这一机制的合理性；它不能替代对当前 checkpoint 的因果实验，也不能证明 LIBERO 数据本身必然破坏语言能力。

## 后续训练怎样区分数据因素与训练目标

要检验训练机制，应从同一个完整 `pi05_base` 初始化，固定 LIBERO 数据、采样顺序和随机种子，比较以下训练条件，并保存中间 checkpoint：

| 条件 | VLM 是否更新 | VLM 文本/token 训练 | 要检验的问题 |
|---|---|---|---|
| 公开动作微调路径 | 更新 | 无 | 文本异常是否随训练出现 |
| 冻结 VLM 的动作微调 | 冻结 | 无 | 同一 LIBERO 数据在冻结 VLM 时是否保留文本输出 |
| 带语言约束的训练 | 更新 | 文本交叉熵或原模型分布约束；可进一步实现 KI | 动作适配能否与文本保持共同完成 |

文本评估需要固定的图像描述、问答和子任务样本，分别统计文本预测损失、可读输出比例和任务正确率；动作评估需要 LIBERO 成功率与原始指令下的对照。若要把原因进一步归于 LIBERO 特有的数据分布，还需要在相同训练目标下加入其他机器人数据的匹配对照。本次没有开展这些训练实验。

若先采用冻结 VLM 的方案，应从 base 重新训练动作专家，让动作专家从训练开始就适配固定的 VLM。当前模型把动作专家也放在 `PaliGemma.llm` 中，因此冻结 `PaliGemma/.*` 会连动作专家一起冻结。可用以下过滤器冻结视觉编码器和 VLM 语言分支，同时保留 `_1` 动作专家分支：

```python
freeze_filter = nnx.All(
    nnx_utils.PathRegex("PaliGemma/.*"),
    nnx.Not(nnx_utils.PathRegex("PaliGemma/llm/.*_1.*")),
)
```

动作输入/输出投影及时间网络位于 `PaliGemma` 之外，也会继续训练。冻结方案仍需实际训练与动作评估；它不会自动修复一个已经生成异常文本的 checkpoint。base 能生成可读文本，也不保证 base 能正确生成 LIBERO 子任务。

## 复现命令

在本次服务器的 openpi 根目录运行：

```bash
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  PYTHONPATH=src:packages/openpi-client/src \
  taskset -c 0-15 /home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python \
  scripts/probe_pi05_text_weight_ablation.py \
  --base /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --libero /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --frame docs/experiments/pi05_first_frame_compare.npz \
  --output docs/experiments/pi05_text_weight_ablation.json
```

实验所依据的源码基线为 `a9846efb8cba555a224aff3db943a7427e9d3ce5`；新增实验脚本与结果随后一并提交。各层差异最多采样每个参数数组每层的 16,384 个值；归一化参数少于该数量时使用全部值。此前第 0 层与第 17 层 Q 的完整差异测量保存在 `docs/experiments/pi05_base_vs_libero_weight_compare.json`。
