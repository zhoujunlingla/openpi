# π0.5 文本探针：第一帧诊断

针对 `zhoujunlingla/openpi` 的提交
`3d8b911ee2dbd657789e50ddc5a5c9fdb86f07fd`。

**这是诊断工具，不是修复后可用的子任务规划器，也不提供新的模型权重。**
它将“缓存/解码数值错误”和“提示、词表、权重的语言能力问题”分开排查，不能单凭数值一致性证明模型会生成正确子任务。

## 安装与启动

保留已有 `debug/pi05-text` 分支的代码。在 openpi 根目录执行：

```bash
# 将本包中的脚本复制到当前 openpi 仓库。这个文件名是新增的。
cp /path/to/pi05_text_audit_v2/serve_pi05_text_audit.py scripts/

uv run scripts/serve_pi05_text_audit.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000 \
  --steps 4 \
  --report data/libero/pi05_text_audit.json
```

沿用当前能够加载的本地 checkpoint 时，将 `--checkpoint` 改为原命令使用的路径。
不要同时在相同端口运行旧服务器。该脚本使用原有 JAX 权重，不支持 PyTorch checkpoint。
第一次请求包含额外 JIT 编译与多次完整 VLM 前向，会明显慢于正常推理。

在已有 LIBERO 客户端环境执行：

```bash
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.text-two-phase \
  --args.text-pause-before-action
```

第一条 plan 请求会生成诊断报告；客户端接着等回车，这时可以先停下，不必运行完整 suite。
若继续，动作仍使用原始任务指令（observe），不使用诊断字符串。

跨机器运行时按原有设置指定服务器 `--host` 和客户端地址。默认仅监听本机。
报告已有同名文件时会拒绝启动；通过 `--report` 指定新文件名，不覆盖原报告。

## 实际检查什么

1. 当前进程导入的 `gemma.py`、`pi0.py`、文本探针与缓存辅助文件路径，以及 Git HEAD。
2. 模型词表维度、SentencePiece 词表维度、BOS/EOS/PAD。
3. 原始 VLM token IDs、token pieces、Unicode 类别，不把 Unicode 转义误认成正常文本。
4. **任何自回归缓存更新之前**，第一 token 的 top-10 logits/概率。
5. 重新计算的首 token 是否与原生产解码器的首 token 一致。
6. 固定同一段 token 历史，逐步比较：
   - 当前分支的 padded-KV-cache 路径；
   - 不使用 KV cache、完整重算 prefix + causal suffix 的参考路径。

参考路径的输入 prefix 是双向 attention；生成后缀是 causal attention；prefix 不能看生成后缀；masked camera/padding 不参与有效 attention。不是把后缀重新当成双向输入。

`--teacher-text` 默认为 `pick up the black bowl`，只用于给两条数值路径提供同一段已知历史。**它是人为输入的测试数据，不是 VLM 生成结果，也不会传入 Action Expert。** 即便原生成结果立即 EOS，这个对照仍然能检查后续缓存更新。

报告包含完整 Unicode 转义形式，便于保留私用区或异常字符；这里没有“修正”原模型输出。

## 如何解释

- `production_first_id_matches_independent_prefill` 为 false：先检查代码/运行环境、非确定性或数值近似，不做语言能力判断。
- 第一 token 已经异常：本实现首 token 尚未经历 pad/compact 更新，不能把这个 token 的异常归咎于后续缓存更新。
- `comparisons` 中两路 top-1 一致且概率分布接近，但原输出无语义：降低对缓存实现错误的怀疑，继续检查提示/词表/权重；不是“已经证明权重完全不会说话”。
- 多步出现显著差异：检查缓存槽位、位置与 attention mask。BF16 不同计算形状会带来数值差异，top-1 margin 很小时单个 argmax 翻转不能单独证明逻辑 bug。
- 两路都能产生可读文字：仍需检查文本是否随图像中的任务阶段变化，并验证 Action Expert 是否能执行新的子任务指令。

不要用强制 ASCII、候选标签约束或复制原任务指令，把“字幕可读”伪装成“恢复了模型的高层推理”。

## 代码接口依据

核对的文件（均为上述指定提交）：

```text
src/openpi/policies/pi05_text_debug.py
src/openpi/policies/pi05_text_cache.py
src/openpi/models/gemma.py
src/openpi/models/pi0.py
src/openpi/policies/policy_config.py
src/openpi/shared/nnx_utils.py
examples/libero/text_debug_visuals.py
```

`TextDecoder.generate` 在构造 padded cache 之前计算第一 token logits。
`gemma.Module` 已经返回 final-norm 后的 hidden states，不应重复加一层 RMSNorm。
`pi0.compute_loss` 的当前实现是动作 flow-matching 损失，没有子任务文本交叉熵。
这些代码事实不等于获知公开 checkpoint 的全部私有训练历史。

## 验证范围

压缩包提供的验证：

- 脚本 `py_compile` 成功。
- `--help` 可正常执行。
- `test_audit.py` 中 9 项独立测试通过：带 holes 的 mask/positions、JAX JIT 与 NumPy 一致性、logits 指标、非法数值处理、Unicode 保留、报告写入、三层合成 attention 的完整重算与增量缓存对照。
- 压缩包作者的环境没有加载真实 π0.5 权重、运行 LIBERO 或执行真实 Flax/NNX 模型接口。
- 合成 attention 测试不是 openpi 模型测试。诊断脚本出错时记录 traceback 并抛出异常，不隐藏失败继续执行动作。

运行独立测试：

```bash
python -m pytest -q test_audit.py
```

诊断成功后最重要的交付物是 `data/libero/pi05_text_audit.json`，而不是新的回放截图。

### L20_node2 的真实首帧结果

在 `pi05_libero` 的本地公开权重上，以 LIBERO Spatial 第一个任务的真实初始观测发送了一次 `plan` 请求，没有发送 `act` 请求。诊断报告的 `status` 为 `completed`。运行进程导入了当前仓库的 `gemma.py`、`pi0.py`、文本探针和缓存辅助文件；SentencePiece 与模型输出词表大小均为 `257152`。

原生成器的首 token ID 与独立 prefix 前向的 argmax 一致。首 token 在任何缓存更新之前已经是不能解释为子任务的字符。`history=0` 的缓存与完整重算路径 top-1 不同，概率总变差约 `0.042`；完整重算路径的前两名 logit 间隔仅 `0.125`，所以这一次排名翻转本身不足以证明缓存逻辑错误。固定 teacher token 历史的后 4 步 top-1 均相同，概率总变差约为 `0.000005` 至 `0.050`。这些差异需要结合 BF16 计算和更多对照继续检查；这份报告不能单独证明权重具备或缺失高层子任务能力。

本次使用的首帧客户端只建立 LIBERO 场景、发送一次 `plan` 请求，并在收到报告后退出。提交 `6d8438e` 对应的复跑报告保存于服务器仓库的 `data/libero/pi05_text_audit_6d8438e.json`；报告中的 teacher 文本没有进入动作策略。
