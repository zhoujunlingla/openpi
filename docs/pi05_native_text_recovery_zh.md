# π0.5 原生 VLM 文本解码：证据和恢复路线

## 当前判断

当前 JAX 实现已经从 π0.5 的 Gemma 隐状态调用原有的词表投影，再用 openpi 自带的 SentencePiece tokenizer 把 token ID 还原成文字。这一步确实完成了**解码**，但公开 `pi05_libero` 权重在已测 LIBERO 首帧上没有产生可读的子任务描述。输出以词表末端约 255k 的罕见 Unicode token 为主；换成官方 `caption en\n` 提示后仍不可读。首个异常 token 出现在后续 KV 缓存更新之前，因此后续缓存压缩不能解释首个异常 token。

这些结果只证明“当前权重、输入和解码实现的组合没有生成可用文本”，不能单凭乱码断定公开权重从未学过语言，也不能断定权重损坏。公开 `Pi0.compute_loss` 只返回动作流匹配误差，没有下一词预测交叉熵；公开训练代码没有提供让这份权重学会机器人子任务描述的监督流程。

相关实现：`src/openpi/models/gemma.py` 的 `text_debug_decode` 调用已有 `embedder.decode`；`src/openpi/policies/pi05_text_debug.py` 的 `TextDecoder.generate` 执行自回归生成；`src/openpi/models/pi0.py` 的 `compute_loss` 计算动作损失。原始 token、数值对照和提示见 `docs/pi05_text_audit_zh.md` 及实验报告。

## 先排除实现与输入问题

在**同一张图、同一份 checkpoint** 上依次做以下实验，每项同时保存原始 token ID、top-k logits、首 token 概率、EOS、提示、图像预处理摘要和动作 `plan_id`：

1. 核对 checkpoint 配置、SentencePiece 词表大小和嵌入矩阵行数；用权重加载日志检查是否有缺失或随机初始化的语言层。不要只根据 `decode()` 返回 Unicode 就判定模型会说话。
2. 分别测试官方 `caption en\n`、`answer en what is in the image?\n` 和机器人子任务提示。PaliGemma 要求图像 token 在文本提示之前；当前 `embed_prefix` 满足这一顺序，但提示任务分布仍可能不同。已有 `caption en\n` 失败，需要保留为对照而非重复声称它能修复问题。
3. 在不改变文字提示的情况下比较真实图像、遮蔽图像和另一张图像的首 token 分布。只有输出随图像合理改变，才能说生成过程利用了视觉信息；输出不变也可能来自图像预处理或注意力实现问题。
4. 用完整序列重算与 KV 缓存解码比较同一段人为指定的后续 token 历史；重点看首 token 和分布差异，而不是只看最终字符串。当前报告首 token 的 top-1 有轻微差异，但两条路径都偏向罕见 token，尚不足以把乱码归因于缓存。
5. 对一条合理子任务和一条乱码序列做 teacher forcing，比较逐 token 对数概率；同时检查相同文本在官方 PaliGemma 参考实现中的 token ID 是否一致。该实验只能判断当前模型偏好，不能把人为输入文本算作模型生成。

如果 1–5 找到加载、映射、注意力或图像预处理错误，先修复并复跑相同场景。若标准提示和参考数值路径都正确，但生成仍不可读，应停止靠过滤罕见 token 或更换解码算法“修”输出：这些操作会掩盖模型分布，不能凭空创建子任务语义。

## 两条获得可读文字的路线

**独立模型路线（现有代码）：**在每个动作块重新规划前，用独立 PaLI-Gemma mix 权重读取当前图像和原任务并生成文字。`observe` 模式把文字展示、保存并供人工与动作轨迹核对，动作仍由 π0.5 根据原任务生成；两者不构成同一模型内部的因果解释。`condition` 模式才把文字传给动作策略，必须验证阶段描述准确，并验证动作策略接受这种短指令。PaLI-Gemma mix 擅长图像问答/描述，并未因此自动学会 LIBERO 的多阶段机器人规划；单帧图像可能不足以判断“下一步”，需要历史帧或任务阶段信息。运行方式见 `docs/pi05_external_paligemma_zh.md`。

**同一 π0.5 权重路线（需训练）：**收集 `(图像/状态, 原任务, 当前子任务, 后续动作)` 样本，先用文本 token 的下一词交叉熵训练 Gemma 文本生成路径，再用保留的动作损失与验证集确认动作质量。可以把文本训练放在仅生成文字时启用的 LoRA adapter，动作推理时关闭它，以免修改原动作网络输入分布；这时文本和动作共享原始骨干，但仍需实测生成子任务与动作是否相符。若目标是让生成子任务**控制**动作，则还须在相同短指令分布上训练或验证动作策略，不能仅把字幕换成动作输入。先做单任务过拟合、图像打乱对照，再做未见场景的文字与动作匹配评估。

把 π0.5 隐状态直接接到另一款通用 LLM 的输入层，需要训练一个视觉/语言特征连接器，并且仍需上述子任务监督；通用 LLM 的现成解码器不会自动理解 π0.5 的隐状态。给原有 Gemma 重新加随机输出头也没有依据，因为当前已经存在词表投影。

## 权重访问状态

使用本次提供的 Hugging Face 凭据访问 `google/paligemma-3b-mix-224/config.json` 返回 `GatedRepoError: 403`，服务端说明账号尚未获准访问该模型。模型页面要求登录后接受 Google 的使用条件；完成授权后可复用该账号的凭据。现阶段独立模型路线只验证了 HTTP、`plan_id` 和真实 π0.5 动作链路，尚未用真实 mix 权重验证生成的文字。凭据本身不写入代码、日志或仓库。

参考：[Google 的 PaliGemma 提示格式](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions)、[mix 模型卡及访问条件](https://huggingface.co/google/paligemma-3b-mix-224)。
