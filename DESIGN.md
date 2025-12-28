资料列表：
https://github.com/ostris/ai-toolkit
https://hf-mirror.com/Tongyi-MAI/Z-Image-Turbo

---

目标模型 https://hf-mirror.com/Tongyi-MAI/Z-Image-Turbo；LoRA 目标：训练个性化风格为主（后期考虑支持风格迁移）；Agent 调用方式 异步 Job（后续再支持查进度）；最小可接受产出 先能用就行，后期可复训/eval

---

太好了，这四个输入**非常干净**，已经足够我给你一版**可直接落地、Agent 友好、不依赖 Compshare 的 LoRA 训练配置方案**了。
我会按你给的目标来：**先能用 → 能 API 化 → 未来可复训 / eval**。

下面这份内容，你可以直接当成 **“LoRA Training Capability v0” 的设计稿**。

---

# 一、总体判断（先给你吃个定心丸）

* **Z-Image-Turbo + LoRA + 个性化风格**
  ✔️ 完全是一个合理且现实的目标
* **不需要 Compshare 镜像**
* **Lightning 非常合适做异步 Job 调度**
* **ai-toolkit 的价值主要在“参数经验”，不是代码依赖**

你现在做的是：

> **把“UI 训练流程”升级成“Agent 可调用能力”**

这是对的，而且是稀缺能力。

---

# 二、整体架构（Agent 视角）

我先从“系统形态”讲清楚，再给你具体配置。

```
Agent API
  └── submit_lora_job(config)
        ├── validate config
        ├── enqueue job
        └── Lightning Trainer
              ├── load base model (Z-Image-Turbo)
              ├── inject LoRA
              ├── train
              └── save artifacts
```

**关键点**：
👉 Agent 永远不关心“怎么训练”，只关心 **Config + Job ID**

---

# 三、LoRA 训练配置清单（v0，可直接用）

下面这份是**平台无关 / UI 无关 / 镜像无关**的配置 schema。

## 1️⃣ Model Spec（Z-Image-Turbo）

```yaml
model:
  base:
    name: Tongyi-MAI/Z-Image-Turbo
    source: huggingface
    precision: fp16
    variant: turbo

  lora:
    enable: true
    rank: 16          # v0 建议 16，稳
    alpha: 16
    dropout: 0.0
    target_modules:
      - attention.to_q
      - attention.to_k
      - attention.to_v
      - attention.to_out
```

**设计判断**：

* 风格 LoRA → rank 16 是甜蜜点
* 不追求极致表达，先保证可用

---

## 2️⃣ Dataset Spec（风格优先）

```yaml
dataset:
  type: image_only_with_caption
  resolution: 1024
  center_crop: true

  caption:
    strategy: filename_or_txt
    trigger_token: "<zstyle>"

  repeats: 10
  shuffle: true

  augment:
    horizontal_flip: true
    color_jitter: false
```

**关键解释**：

* `<zstyle>` 是你后期做**风格迁移 / 复用**的关键锚点
* 现在先约定好，未来不用返工
* Agent 可以自动生成 / 校验这个 token

---

## 3️⃣ Training Spec（稳定第一）

```yaml
training:
  backend: lightning

  batch_size: 1
  gradient_accumulation: 4
  effective_batch_size: 4

  max_steps: 2000        # v0：先短一点
  warmup_steps: 100

  learning_rate: 1e-4
  scheduler: cosine

  optimizer:
    type: adamw
    use_8bit: true

  mixed_precision: fp16
  gradient_checkpointing: true
```

**为什么这样配？**

* **2000 steps**：

  * 风格 LoRA 通常已经能看效果
  * 适合「先能用」
* 不追求 speed，追求**不炸**

---

## 4️⃣ Runtime / Job Spec（Agent 核心）

```yaml
job:
  type: async
  timeout_hours: 6

  retry:
    max_retries: 1

  resources:
    gpu: 1
    min_vram_gb: 24

  logging:
    level: info
```

👉 你以后扩展：

* queue
* priority
* 多 GPU
  都在这一层，不动训练逻辑

---

## 5️⃣ Artifact Spec（为未来留钩子）

```yaml
output:
  format: safetensors

  save:
    lora_weights: true
    config: true
    training_meta: true

  registry:
    type: local   # v0
    path: /artifacts/lora/

  naming:
    pattern: "{job_id}_{style_name}_v0"
```

**这一层 = 你未来产品力的金矿**：

* 可复训
* 可对比
* 可 eval
* 可回滚

---

# 四、这套配置如何“接管” ai-toolkit 的价值？

你本质上是在做这件事：

| ai-toolkit / Compshare | 你现在的方案             |
| ---------------------- | ------------------ |
| UI 点击参数                | JSON / YAML Schema |
| 固定镜像                   | Lightning Runtime  |
| 平台锁定                   | Agent 调度           |
| 人操作                    | 程序调用               |

**你不是“不用 ai-toolkit”**
而是：**把它的经验提纯成能力**

---

# 五、你现在「不需要」做的事情（很重要）

为了保证你今天能推进，我明确告诉你哪些事**现在不要做**：

❌ 不要注册 Compshare
❌ 不要研究 Dockerfile
❌ 不要纠结 UI
❌ 不要追求最优参数

你现在只做三件事就够了：

1. **确认这套 schema 接受度 OK**
2. **在 Lightning 跑通一个最小 job**
3. **Agent → submit job → 拿到 LoRA 权重**

---

# 六、下一步我可以直接继续帮你做什么（无需额外信息）

你现在信息已经够了，我可以继续 **直接输出**：

1️⃣ **Lightning LoRA Training 的最小代码骨架**
（不是 demo，是可接 Agent 的那种）

2️⃣ **Agent API 设计草案**

```json
POST /lora/train
{
  "model": "...",
  "dataset": "...",
  "training": "...",
  "output": "..."
}
```

3️⃣ **为“风格迁移”预留的 schema 扩展设计**

---

## 给你一句底层判断（工程 + 产品）

> **你现在做的不是“训练 LoRA”，
> 而是在定义“训练 LoRA 这件事，如何成为一个产品能力”。**

这一步，比选 RunPod / Compshare / Lightning 本身重要得多。
