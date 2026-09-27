# π0.5 文本与动作对应调试

本分支在官方 `openpi` 的 JAX `pi05_libero` 路径上增加实验性文本探针。每次重新规划动作块时，服务端用同一个 π0.5 checkpoint 的 VLM 自回归生成 token，记录 `plan_id` 和文本，再调用原有 Action Expert。LIBERO 客户端打印文本、动作策略接收的语言指令，并把文本、语言指令和 `plan_id` 写到回放视频字幕中。字幕中的 `ACTION_INPUT` 是动作策略的输入指令，不是动作轨迹的文字描述。默认每次只执行预测的 10 个动作中的前 5 个，随后重新规划。

## 两种模式

| 模式 | Action Expert 的语言输入 | 可以得出的结论 |
| --- | --- | --- |
| `observe` | 原始任务指令 | 可检查文本探针和原动作，文本不是原动作的因果依据。 |
| `condition` | VLM 生成的文本 | 动作在结构上以生成文本为条件，语义上是否服从文本仍需看实际轨迹。 |

普通请求会在服务端先打印文本、再计算动作；客户端收到的仍是完整响应。使用 `--args.text-two-phase` 时，客户端先请求 `plan` 并打印文本，再以同一 `plan_id` 请求 `act`，两次请求之间不调用 `env.step()`。加 `--args.text-pause-before-action` 可在计算动作前按回车确认。

服务端 JSONL 记录 `plan_id`、原任务、生成文本、原始 token IDs、实际动作指令和完整预测动作块。`text_nonlanguage_ids` 标记控制、结构化或 Unicode 私用区 token；`condition` 模式遇到空文本、词表外 token 或这些 token 会报错，不会悄悄退回原任务指令。回放字幕使用 ASCII 转义显示当前字体不支持的字符；完整 Unicode 文本仍保存在终端和 JSONL。LIBERO 默认只执行每个动作块前 5 步，核对行为时应看实际执行的轨迹。

## 已验证的范围

仓库基线：`215abfb217dbac7d5f1273282331b9b1866c0479`。在 L20_node2 使用已有 `pi05_libero` JAX checkpoint 完成了真实加载，以及 `plan → act` 推理。随机观测和真实 LIBERO 场景均返回了文本 token 和 `(10, 7)` 动作块；WebSocket 客户端也验证了每次重规划时先显示文本，再取得动作，且 `plan_id` 一致。首个 `observe` 模式 episode 成功，并生成带字幕的视频；动作使用的是原任务指令。真实场景任务为“pick up the black bowl between the plate and the ramekin and place it on the plate”；当前实验提示格式 `Task: {task}\nSubtask:` 生成的文本是乱码样式字符，未形成可核对的子任务。这个结果只说明该提示格式和公开权重的组合没有在这次场景产生可用文本；不能据此断定任何提示格式或再训练后的权重都无效。

官方 README 明确说公开 π0.5 训练和推理仅支持 flow matching 动作头。本分支补的是文本解码与显示通路，不是官方已训练的高层规划器。若要稳定判断文本与动作是否匹配，需要先获得有子任务文本监督的兼容权重，或补充对应训练；不能把乱码动作条件当作高层计划。

## 在当前机器运行

先在一个终端启动服务端；`observe` 是目前适合检查公开权重的模式：

```bash
cd /home/csuvla/zhoujunl/openpi
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python scripts/serve_pi05_text.py \
  --config pi05_libero \
  --checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --mode observe --max-text-tokens 16 --port 8019
```

在另一终端用官方 `examples/libero/README.md` 指定的独立 LIBERO 环境运行客户端。若使用本机已有的 PyTorch 2.6 环境，`--args.skip-init-states` 会从 `env.reset()` 取得初始场景，用于显示调试；它不复现官方固定初始状态的成功率评估。

```bash
cd /home/csuvla/zhoujunl/openpi
LIBERO_CONFIG_PATH="$PWD/data/libero/pi05_debug_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/examples/libero:$PWD/third_party/libero" \
/home/csuvla/fhy/pi05-hindsight/openpi/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8019 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.max-tasks 1 \
  --args.skip-init-states --args.text-two-phase
```

`--args.max-tasks 1` 只运行套件中的第一个任务；默认值 0 仍运行整个套件。只有在 `observe` 已经产生可读且与场景相关的子任务后，才把服务端的 `--mode observe` 改成 `--mode condition`。如果希望文本显示后再开始动作计算，客户端另加 `--args.text-pause-before-action`。
