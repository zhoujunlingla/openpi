# `lerobot/pi05_libero_base` 文本探测

## 结论

这份 Hugging Face 权重与 OpenPI 的 `pi05_libero` 微调权重高度吻合，不能把仓库名里的 `base` 理解为 OpenPI 的 `pi05_base`。在同一帧 LIBERO 图像上，用这份权重的 VLM（视觉语言模型）连续生成 16 个 token，`caption en`、图像问答和任务描述三种提示都没有产生可读的任务文字。三次生成均未遇到结束 token。

这只说明本次图像和提示下的实验性文字探测失败。它不等于 LeRobot 官方动作推理失败，也不是对所有图像的文字能力作保证。官方[模型卡](https://huggingface.co/lerobot/pi05_libero_base)说明目前只提供流匹配动作头，没有发布子任务预测接口。

## 权重和配置核对

- Hugging Face 仓库：`lerobot/pi05_libero_base`，revision `a217bfd3b14673cf2ce597e69997ab21866438dd`。
- `model.safetensors`：14,467,165,872 字节；SHA-256 `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4`，与仓库公布的值一致。
- 从语言模型第 0 层 `q_proj` 的一个完整输出行取 2,048 个参数，与 OpenPI JAX 权重对应行比较：对 `pi05_libero` 的均方根误差为 `3.89e-5`，对 `pi05_base` 为 `4.28e-3`。这是它来自 LIBERO 微调权重的强证据；完整数值见 [权重对比](experiments/pi05_lerobot_libero_base_weight_compare.json)。转换时的舍入使两个文件并非逐位相同。
- Hugging Face 配置的 `chunk_size` 为 50；此前 OpenPI `pi05_libero` 的 `action_horizon` 为 10。即使权重来源相同，也不能把两套配置称为完全相同的推理程序。
- LeRobot 加载时报告 812 个状态字典键全部成功加载；探测脚本另行核对了模型内 `q_proj` 参数与下载文件的数值。

## 同一帧图像的结果

任务原文：`pick up the black bowl between the plate and the ramekin and place it on the plate`。图像、手腕相机图像和任务文字取自 [保存的首帧](experiments/pi05_first_frame_compare.npz)，文件 SHA-256 为 `2527af8d5fa09247dd84d8b198054623fe9b907ac70eb249d29f742ec1eb6813`。

| 提示 | 前几个生成 token ID | 解码文本开头 | 16 token 内结束 |
| --- | --- | --- | --- |
| `caption en` | `3351, 255689, 255693, 255695` | `SubẀ⊏ⓙ…` | 否 |
| `answer en what objects are visible in the image?` | `254412, 255171, 255503, 255705` | `峫ㆆ…` | 否 |
| `Task: …\nSubtask:` | `255629, 254899, 254899, 254899` | `뗄…` | 否 |

多数 token 落在词表末端，解码后是罕见字符或私有区字符。这与此前用 OpenPI `pi05_libero` 做 JAX 探测时看到的乱码现象一致，但两套实现的逐 token 输出并不相同。完整提示、token ID、文本和结束标志见 [实验结果](experiments/pi05_lerobot_libero_base_text_probe.json)。

## 复现范围

使用 LeRobot 源码提交 `e595b7902714ba51f91e47523f66f89c5181b649`、Python 3.12、PyTorch 2.8.0、Transformers 5.5.4。脚本是 [probe_lerobot_pi05_text.py](../scripts/probe_lerobot_pi05_text.py)。它读取 OpenPI 使用的同一 SentencePiece 词表，将 VLM 最后一层隐藏状态投影到词表并逐 token 贪心解码；没有让动作专家生成文字。该脚本不调用 LIBERO 动作评估，因此本实验没有成功率数字。

```bash
PYTHONPATH=/path/to/lerobot/src python scripts/probe_lerobot_pi05_text.py \
  --model /path/to/pi05_libero_base \
  --frame docs/experiments/pi05_first_frame_compare.npz \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/pi05_lerobot_libero_base_text_probe.json \
  --device cuda:0 --max-tokens 16
```

脚本会先核对权重文件 SHA-256，再加载 LeRobot 模型并检查已加载参数。模型输出的文字仅作为诊断记录；没有送入动作计算。
