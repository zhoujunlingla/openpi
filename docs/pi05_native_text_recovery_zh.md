# π0.5 原生 VLM 文本解码：证据和恢复路线

## 为什么 Action Expert 不能直接输出文本

`pi05_libero` 的 Action Expert 使用流匹配：`action_out_proj` 把隐藏向量映射成每步 32 个连续数值，经过多步积分得到 `(10, 32)` 动作块；LIBERO 输出变换只取每步前 7 维。它没有从动作隐藏向量映射到词表的文本输出头，也不按文本 token 训练。直接把动作数值送入 SentencePiece、对动作向量取 `argmax`，都不能得到模型生成的任务描述。π0-FAST 虽然预测离散 token，但这些 token 编码动作，属于另一种模型及训练目标。

文本出口是另一条路径：当前 JAX 探针从 π0.5 的 Gemma 隐状态调用已有的词表投影，再用 openpi 自带的 SentencePiece tokenizer 把 token ID 还原成文字。这一步确实完成了**解码**；问题是 `pi05_libero` 权重没有在已测 LIBERO 首帧生成可读文本。

代码依据：`src/openpi/models/pi0.py` 的 `action_out_proj`、`sample_actions` 与 `compute_loss`；`src/openpi/models/gemma.py` 的 `text_debug_decode`；`src/openpi/policies/pi05_text_debug.py` 的 `TextDecoder.generate`。公开 `Pi0.compute_loss` 只返回动作流匹配误差，没有下一词预测交叉熵；这不能反推出未公开的全部训练历史。

## 同一画面、两份权重的实测结果

使用 LIBERO Spatial 第一个任务的同一初始观测、同一 `pi05_libero` 模型配置、同一 tokenizer、同一解码代码和相同的 LIBERO 归一化统计，分别加载 `pi05_base` 与 `pi05_libero`：

| 提示 | `pi05_base` | `pi05_libero` |
| --- | --- | --- |
| `caption en\n` | 可读描述，以 `The image shows a collection of objects...` 开头 | 16 个以词表末端约 255k 为主的罕见字符 |
| `answer en what objects are visible in the image?\n` | 可读回答，以 `In the image, there are several objects...` 开头 | 罕见字符，随后 EOS |
| `Task: ...\nSubtask:\n` | `pick up small plate` | 罕见字符，无 EOS |

基座的 `pick up small plate` **没有准确描述原任务中的黑碗**；“可读”不能当作“子任务正确”。但基座能用相同代码输出英语，使“词表映射或解码算法普遍坏了”不再是主要解释。已有单权重审计还表明，`pi05_libero` 的首个异常 token 出现在任何后续 KV 缓存更新之前；后续缓存压缩不能解释首 token。

权重对照进一步显示：乱码首 token ID `255684` 的词嵌入行在两份权重中完全相同；抽样的词表末端 1000 行相对 RMS 变化约 `0.000065`，Gemma 最终归一化约 `0.000138`。Gemma 第 0 层注意力 Q 权重变化约 `0.198`，第 0 层 MLP 抽样变化约 `0.147`。这些数值支持“LIBERO 权重中的 Gemma 内部表示发生变化，导致原有词表投影不再给出可读文字”；它们没有证明是哪个单独参数造成了乱码。公开 `pi05_libero` 训练配置从 `pi05_base` 加载权重，随后采用仅有动作损失的训练代码，因此动作微调导致语言生成退化是目前最有证据的解释。

原始结果：[文本对照](experiments/pi05_base_vs_libero_text_compare.json)、[参数对照](experiments/pi05_base_vs_libero_weight_compare.json)、[先前缓存审计](pi05_text_audit_zh.md)。

## 复跑对照

在 `L20_node2` 的仓库根目录，先用 LIBERO 环境保存一次观测，再用 JAX 环境比较两份已下载权重。三个脚本均不运行 Action Expert 文本解码，也不会修改 checkpoint。

```bash
LIBERO_CONFIG_PATH="$PWD/data/libero/pi05_debug_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/examples/libero:$PWD/third_party/libero" \
/home/csuvla/fhy/pi05-hindsight/openpi/.venv/bin/python \
  scripts/debug_pi05_save_libero_frame.py data/libero/pi05_first_frame_compare.npz

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python \
  scripts/debug_pi05_compare_checkpoints.py \
  --frame data/libero/pi05_first_frame_compare.npz \
  --checkpoint-root /home/csuvla/.cache/openpi/openpi-assets/checkpoints \
  --output data/libero/pi05_base_vs_libero_text_compare.json

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python \
  scripts/debug_pi05_compare_weights.py \
  --checkpoint-root /home/csuvla/.cache/openpi/openpi-assets/checkpoints \
  --output data/libero/pi05_base_vs_libero_weight_compare.json
```

如果需要继续定位，下一组对照应比较同一提示下不同图像的首 token 分布、逐层 Gemma 隐状态差异，以及合理子任务和乱码的 teacher-forced 对数概率。不要用强制 ASCII 或过滤词表末端 token，把其他 token 的输出冒充为模型恢复的任务描述。

## 两条获得可读文字的路线

**独立模型路线（现有代码）：**在每个动作块重新规划前，用独立 PaLI-Gemma mix 权重读取当前图像和原任务并生成文字。`observe` 模式把文字展示、保存并供人工与动作轨迹核对，动作仍由 π0.5 根据原任务生成；两者不构成同一模型内部的因果解释。`condition` 模式才把文字传给动作策略，必须验证阶段描述准确，并验证动作策略接受这种短指令。PaLI-Gemma mix 擅长图像问答/描述，并未因此自动学会 LIBERO 的多阶段机器人规划；单帧图像可能不足以判断“下一步”，需要历史帧或任务阶段信息。运行方式见 `docs/pi05_external_paligemma_zh.md`。已下载的 `pi05_base` 也可作为独立文字探针；将它的文字接入 `pi05_libero` 动作策略后，[一次 LIBERO 回合成功](pi05_base_text_libero_condition_zh.md)，但首段文字说错目标。仅用 `pi05_base` 同时生成文字和动作的 [30 回合评估为 0/30](pi05_base_only_libero_spatial_zh.md)。两项结果说明文字是否可读与动作策略能否执行是两个必须分别检验的问题。

**同一 π0.5 权重路线（需训练）：**收集 `(图像/状态, 原任务, 当前子任务, 后续动作)` 样本，先用文本 token 的下一词交叉熵训练 Gemma 文本生成路径，再用保留的动作损失与验证集确认动作质量。可以把文本训练放在仅生成文字时启用的 LoRA adapter，动作推理时关闭它，以免修改原动作网络输入分布；这时文本和动作共享原始骨干，但仍需实测生成子任务与动作是否相符。若目标是让生成子任务**控制**动作，则还须在相同短指令分布上训练或验证动作策略，不能仅把字幕换成动作输入。先做单任务过拟合、图像打乱对照，再做未见场景的文字与动作匹配评估。

把 π0.5 隐状态直接接到另一款通用 LLM 的输入层，需要训练一个视觉/语言特征连接器，并且仍需上述子任务监督；通用 LLM 的现成解码器不会自动理解 π0.5 的隐状态。给原有 Gemma 重新加随机输出头也没有依据，因为当前已经存在词表投影。

## 权重访问状态

使用本次提供的 Hugging Face 凭据访问 `google/paligemma-3b-mix-224/config.json` 返回 `GatedRepoError: 403`，服务端说明账号尚未获准访问该模型。模型页面要求登录后接受 Google 的使用条件；完成授权后可复用该账号的凭据。现阶段独立模型路线只验证了 HTTP、`plan_id` 和真实 π0.5 动作链路，尚未用真实 mix 权重验证生成的文字。凭据本身不写入代码、日志或仓库。

参考：[Google 的 PaliGemma 提示格式](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions)、[mix 模型卡及访问条件](https://huggingface.co/google/paligemma-3b-mix-224)。
