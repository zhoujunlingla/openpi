# 每次重规划时用独立 PaLI-Gemma 生成文字

`serve_pi05_external_text.py` 在每个动作块推理前调用独立的 PaLI-Gemma mix 服务，记录文字、原始 token IDs、模型来源和 `plan_id`，然后调用原有 π0.5 Action Expert。LIBERO 客户端默认每执行 5 个动作重新规划一次；这里的“每次”指每次重新规划动作块，不是模拟器的每一步。

这套文字来自**另一份 PaLI-Gemma 权重**，不是公开 π0.5 checkpoint 的原生高层输出。`observe` 模式下，Action Expert 继续接收原始任务指令，生成文字只展示和记录。`condition` 模式会把生成文字作为 Action Expert 的语言输入，但必须先分别验证文字是否描述当前阶段，以及原动作权重能否执行这种短指令。服务端不会把空文字或请求错误静默替换成原始任务。

## 权重与运行

默认模型是 Google 的 `google/paligemma-3b-mix-224`。它需要在 Hugging Face 模型页面接受使用条件，并以获授权的账号下载；也可以给 `--model` 指定已获授权的本地完整模型目录。模型的 mix 版本可直接用于图像问答和描述，但默认提示“下一步该做什么”仍是实验提示，不能预设它已学会机器人子任务。当前 L20_node2 的 Hugging Face 账号访问该模型返回 403；在获得权重前，下列真实文本生成无法验证。

分别使用两张 GPU，以免 JAX 动作模型与 PyTorch VLM 抢占同一张卡。先启动文字服务：

```bash
cd /home/csuvla/zhoujunl/openpi
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python scripts/serve_paligemma_text.py \
  --model /path/to/authorized/paligemma-3b-mix-224 \
  --device cuda:0 --port 8020
```

已获得 Hugging Face 访问权限并能联网时，`--model` 可改为 `google/paligemma-3b-mix-224`。默认提示为 `answer en what should the robot do next to complete this task: {task}?\n`，可用 `--prompt-template` 修改。提示由模型服务记录在返回值 `high_level_query` 中。

再启动动作服务：

```bash
cd /home/csuvla/zhoujunl/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src" \
/home/csuvla/ysc/workspace/pi0_clean/.venv/bin/python scripts/serve_pi05_external_text.py \
  --config pi05_libero \
  --checkpoint /home/csuvla/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --text-service-url http://127.0.0.1:8020/generate \
  --mode observe --port 8019
```

LIBERO 客户端沿用原有两阶段请求，文字先返回，再发出同一 `plan_id` 的动作请求：

```bash
cd /home/csuvla/zhoujunl/openpi
LIBERO_CONFIG_PATH="$PWD/data/libero/pi05_debug_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/examples/libero:$PWD/third_party/libero" \
/home/csuvla/fhy/pi05-hindsight/openpi/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8019 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.max-tasks 1 \
  --args.skip-init-states --args.text-two-phase
```

终端和回放字幕会标明 `VLM_SOURCE`；`ACTION_INPUT` 始终表示 Action Expert 实际收到的语言指令，不是从动作轨迹推断出的文字。动作服务的 `data/libero/pi05_external_text_trace.jsonl` 保存每次重规划的 VLM 文字、模型名、token IDs、动作输入和预测动作块。只有实际场景测试确认文字可读、随观测阶段改变，才能进一步判断文字与动作是否匹配。

## 当前验证范围

代码中的 HTTP 请求、图像传输、`observe`/`condition` 动作输入选择，以及非法文字阻止动作的逻辑通过了 5 项无权重测试。还用明确标记为 `TEST_STUB_NO_VLM_WEIGHTS` 的本地测试服务和真实 π0.5 权重完成一次 LIBERO 首帧 `plan → act`：动作块形状为 `(10, 7)`，`ACTION_INPUT` 是原始任务，`plan_id` 在文字和动作结果中一致。测试文字由测试服务写死，不能当作 VLM 输出。真实 PaLI-Gemma mix 权重尚未加载，因此目前没有证据表明默认提示会生成正确的 LIBERO 子任务。

参考：[Google 的 PaLI-Gemma 提示格式](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions)、[官方 mix 模型卡与使用条件](https://huggingface.co/google/paligemma-3b-mix-224)。
