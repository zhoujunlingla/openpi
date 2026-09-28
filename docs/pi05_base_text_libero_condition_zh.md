# LIBERO 连续推理：pi05_base 生成文字，pi05_libero 根据文字输出动作

`serve_pi05_base_text_libero_actions.py` 同时加载两份 JAX 权重。每次重规划动作块时，`pi05_base` 先读当前图像和原任务，生成短句；`pi05_libero` 再把**这段短句作为语言输入**预测动作。两份权重分开加载，VLM 文字的来源标为 `pi05_base_checkpoint_vlm_probe`。这不是 `pi05_libero` 自己生成的高层解释，也不是 Action Expert 把连续动作解码成文字。

`condition` 模式的原任务只用于生成文字，动作策略收到 `action_prompt=subtask`。空文字、词表外 token 或非语言 token 会中止当前请求，不会悄悄把原任务送给动作策略。`observe` 模式可作为对照：展示同样的文字，动作策略仍接收原任务。

## 已完成的一次真实 LIBERO 回合

- 任务：LIBERO Spatial 的第一个任务，将盘子和小碟之间的黑碗放到盘子上；随机种子 7，`--skip-init-states`，一个回合。
- 推理：执行了 16 次 `plan → act`，每次预测 `(10, 7)` 动作，只执行前 5 步后重新观测；回放有 76 帧，模拟器报告该回合成功。
- 路由：16 条记录的 `plan_id` 各不相同，`action_prompt` 全部与对应 `subtask` 完全相同；动作数组均为有限值。
- 文本：第 1 段是错误的 `pick up small plate`；第 2–11 段主要是 `pick up the black bowl` 及近似说法；第 12–16 段转为把碗放上盘子，其中第 12、16 段说 `metal bowl`，与原任务的 `black bowl` 描述不同。

这个回合表明两份公开权重可以按“基座文字 → LIBERO 动作”在同一个模拟任务中连续推理并成功一次。一次成功不能证明文字总是准确，也不能证明成功依赖于生成文字；要判断相对收益，需在**相同初始状态和随机种子**下比较 `condition` 与原任务直达动作策略的多回合结果。生成文字和动作的逐步对照也仍需人工检查。

结果文件：[16 次重规划的完整 JSONL](experiments/pi05_base_to_libero_condition_trace.jsonl)、[带文字和动作输入字幕的回放](experiments/pi05_base_text_libero_condition_success.mp4)。JSONL 保存完整预测动作块；模拟器每次只执行前 5 步。

## 在 L20_node2 复跑

先在仓库根目录启动服务；两份权重与 LIBERO 归一化统计已下载到下面的路径。示例把两个 JAX 模型放在同一张空闲 GPU 上。

```bash
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python \
  scripts/serve_pi05_base_text_libero_actions.py \
  --text-checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --action-checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --mode condition --max-text-tokens 16 --port 8022 \
  --log-jsonl data/libero/pi05_base_to_libero_condition_trace.jsonl
```

另开终端运行客户端：

```bash
LIBERO_CONFIG_PATH="$PWD/data/libero/pi05_debug_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/examples/libero:$PWD/third_party/libero" \
/home/csuvla/fhy/pi05-hindsight/openpi/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8022 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 \
  --args.max-tasks 1 --args.skip-init-states --args.text-two-phase \
  --args.video-out-path data/libero/videos_base_condition
```

终端先显示 `[VLM BEFORE ACTION]`，随后显示 `[LOW-LEVEL PROMPT]`；回放字幕中的 `VLM_SOURCE` 是文字权重来源，`ACTION_INPUT` 是动作策略实际收到的指令。重复运行时另设 `--log-jsonl` 与 `--args.video-out-path`，避免新旧回合混在一起。
