# 只用 pi05_base 推理 LIBERO：文字与成功率

本次实验只加载 `pi05_base` 模型权重：同一份权重先从当前画面和原任务生成文字，再根据当前画面和**生成的文字**预测动作。`--mode condition` 确保文字确实成为 Action Expert 的语言输入。`pi05_libero` 目录只提供 LIBERO 状态、动作的归一化统计文件；没有加载其中的模型参数。这个统计映射是让基座动作数值接入 LIBERO 环境所必需的，但它不能替代 LIBERO 动作微调。

## 协议与结果

| 项目 | 结果 |
| --- | --- |
| 环境 | LIBERO Spatial 全部 10 个任务 |
| 回合 | 每任务 3 回合，共 30 回合；随机种子 7，`--skip-init-states` |
| 动作循环 | 每次预测 `(10, 7)`，执行前 5 步后重新观测并生成文字 |
| 成功 | **0/30，观察成功率 0%**；10 个任务各为 0/3 |
| 重规划 | 1320 次，每次 `action_prompt` 均与该次 `subtask` 完全相同，动作形状正常且数值有限 |
| 文字 | 1274 条以 `pick` 开头，5 条以 `put` 开头；也出现 `pick up small plate`、`pick up the blue plate`、`No, not visible` 等不适合作为当前子任务的文字；22 条达到 16 token 上限，可能被截断 |

模拟器客户端正常退出并报告 `Total success rate: 0.0`、`Total episodes: 30`。所有回合均执行到任务步数上限，没有由于空文字或 token 检查错误提前终止。第一任务首回合的回放显示黑碗一直未被抓起，机械臂逐渐偏离目标；这只是一个回合的目视观察，不能据此归因所有失败。

完整统计：[逐任务结果和常见文字](experiments/pi05_base_only_spatial_summary.json)。[逐次文字 CSV](experiments/pi05_base_only_spatial_texts.csv)便于直接检查 1320 次输出和实际动作指令；[完整预测动作块](experiments/pi05_base_only_spatial_trace.jsonl.gz)保存在 gzip 压缩的 JSONL 中。[首任务首回合失败回放](experiments/pi05_base_only_spatial_task0_failure.mp4)带有 VLM 文字与实际动作输入字幕。

这个 **0/30 只适用于本实验的 `condition` 模式和归一化接法**。它不能单独证明失败是由文字错误造成，还是由于未经 LIBERO 微调的基座 Action Expert 不适合该场景。上一组双权重实验用 `pi05_base` 文字和 `pi05_libero` 动作权重在第一个任务中成功了 1/1，见 [双权重回合](pi05_base_text_libero_condition_zh.md)；两个结果的样本数不同，不能直接当作可靠的成功率差异。

## 在 L20_node2 复跑

在仓库根目录启动服务：

```bash
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python \
  scripts/serve_pi05_text.py --config pi05_libero \
  --checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --norm-stats-checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --mode condition --max-text-tokens 16 --port 8023 \
  --log-jsonl data/libero/pi05_base_only_spatial_trace.jsonl
```

另开终端运行客户端：

```bash
LIBERO_CONFIG_PATH="$PWD/data/libero/pi05_debug_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/examples/libero:$PWD/third_party/libero" \
/home/csuvla/fhy/pi05-hindsight/openpi/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8023 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 3 \
  --args.max-tasks 10 --args.skip-init-states --args.text-two-phase \
  --args.video-out-path data/libero/videos_base_only_spatial
```

重复实验时另设 JSONL 和视频目录，避免把多次运行混在一个成功率里。
