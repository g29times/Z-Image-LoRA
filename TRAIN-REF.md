完整 LoRA 训练指南
什么是 LoRA 训练?
LoRA(低秩适应) 允许您以最少的计算资源微调 Z-Image Turbo 以生成特定角色、风格或主题。本节提供创建真实人物 LoRA 的权威指南。

训练概述
方面	规格
数据集大小	70-80 张高质量照片
训练时间	30-40 分钟 (RTX 5090)
所需显存	24GB(使用优化可在 16GB 上运行)
总步数	4000 步
Linear Rank	64(对皮肤纹理至关重要)
工具	AI Toolkit(本地或 RunPod)
步骤 1:收集训练照片
照片要求
数量: 最少 70-80 张图像

质量分布:

高质量特写: 40-50%(面部细节、表情)
中景: 30-40%(上半身、不同角度)
全身照: 10-20%(姿势、服装)
多样性检查清单:

✅ 多个角度(正面、侧面、3/4 视角)
✅ 各种表情
✅ 不同光照条件
✅ 多套服装(如适用)
✅ 自然和摆拍照片
⚠️ 质量影响
数据集的质量直接决定输出质量。颗粒感输入照片将产生颗粒感生成结果。干净的高分辨率图像产生专业结果。

步骤 2:数据集清理
基本清理步骤
移除不需要的元素:

水印和文字叠加
画面中的其他人
分散注意力的背景(如有必要)
裁剪和重新构图:

聚焦于主体
使用一致的构图
移除过多的空白空间
标准化分辨率:

以最长边 1024 像素导出
保持纵横比
使用高质量导出设置
推荐工具
Adobe Lightroom - 专业批处理
Windows 照片 - 快速裁剪
Topaz Photo AI - 质量增强(可选)
步骤 3:可选质量增强
对于低质量源图像:

Topaz Photo AI 设置:

启用仅面部增强以避免伪影
避免全图增强(可能产生塑料感头发)
使用适度锐化设置
保留自然皮肤纹理
💡 专业提示
仅增强真正低质量的图像。过度处理会引入模型将学习和复制的不自然伪影。

步骤 4:数据集标注
命名约定
简单有效的方法:

a photo of [主体名称]
对于不寻常元素:

a photo of [主体名称] with [特定特征]
示例:

✅ "a photo of Wednesday"
✅ "a photo of Wednesday with ponytail"
✅ "a photo of Wednesday without face"(仅身体照片)
✅ 最佳实践
保持标注简单。模型将自动学习一致特征(如标志性服装),无需明确标记。

步骤 5:AI Toolkit 配置
训练参数
# 核心设置
model: Tongyi-MAI/Z-Image-Turbo
training_adapter: V2 (必需)
trigger_word: none (不必要)

# 性能设置
low_vram: false (RTX 5090 禁用)
quantization_transformer: none (强大 GPU)
quantization_text_encoder: none (强大 GPU)

# 对于性能较弱的 GPU:
# quantization_transformer: float8
# quantization_text_encoder: float8

# LoRA 配置
linear_rank: 64  # 对真实皮肤纹理至关重要
# 不要使用 16 或 32 - 结果会很差

# 训练计划
total_steps: 4000
save_every: 每 250 步
checkpoints_to_keep: 6-7 (步骤 2500-4000)

# 优化器设置
optimizer: adam8bit
learning_rate: 0.0002
weight_decay: 0.0001
timestep_type: sigmoid  # 重要!

# 数据集设置
training_resolution: 512  # 更高分辨率不会增加太多好处
sample_generation: false  # 禁用以节省时间
可视化配置参考
工作流配置
示例:完整的 AI Toolkit 工作流配置

步骤 6:训练过程
时间线
步骤 0-1000:    初始学习(不可用)
步骤 1000-2000: 基本特征出现
步骤 2000-3000: 达到可用质量
步骤 3000-4000: 最佳点 - 最优平衡
步骤 4000+:     过拟合风险
检查点选择
推荐保存的检查点:

步骤 2500(早期选项)
步骤 2750
步骤 3000(通常不错)
步骤 3250
步骤 3500(通常最优)
步骤 3750
步骤 4000(最终)
💡 测试策略
使用每个检查点生成测试图像,以找到准确性和灵活性之间的最佳平衡。

步骤 7:使用您的 LoRA
生成设置
# 加载您的 LoRA
pipe.load_lora_weights("path/to/your_lora.safetensors")

# 生成参数
prompt = "a photo of [主体名称], [所需场景/动作]"
num_inference_steps = 9
guidance_scale = 0.0  # Turbo 模型保持为 0
lora_scale = 0.7-1.0  # 调整强度
示例提示词
# 基本生成
"a photo of Merlina, professional portrait, studio lighting"

# 带有标志性服装
"a photo of Merlina, school uniform, outdoor setting"

# 创意场景
"a photo of Merlina, wearing elegant evening dress, at gala event"
训练结果示例
示例 1
示例 2
示例 3
示例:高质量 LoRA 生成结果展示一致的角色特征

最佳实践和技巧
图像生成
提示词工程
✅ 应该:

使用详细、描述性的提示词
指定光照和氛围
包含风格关键词(照片级真实感、电影感等)
利用双语能力生成中文文本
❌ 不应该:

使用极短的提示词(除非有意为之)
仅依赖负面提示词
在 Turbo 模型中使用 guidance_scale > 0
硬件优化
GPU	推荐设置
RTX 4090/5090	bfloat16, Flash Attention, 无 CPU 卸载
RTX 4080/4070 Ti	bfloat16, 如需要则 CPU 卸载
RTX 4060 Ti 16GB	float8 量化, CPU 卸载
RTX 3090	bfloat16, 适度批量大小
LoRA 训练
数据集质量检查清单

收集 70-80 张高质量图像

移除水印和文字

图像裁剪和重新构图

分辨率标准化为 1024px 最长边

包含多样化角度和表情

应用简单、一致的标注
训练优化
更快训练:

使用带 RTX 5090 的 RunPod
禁用样本生成
使用 float8 量化(略微质量权衡)
最佳质量:

使用 Linear Rank 64
训练完整 4000 步
不使用量化(如显存允许)
测试多个检查点
常见问题和解决方案
问题	解决方案
颗粒感输出	使用更高质量的训练图像
过拟合	使用较早的检查点(3000-3500 步)
面部细节差	增加数据集中的面部特写
特征不一致	在训练数据中添加更多样化角度
显存错误	启用 CPU 卸载或使用 float8 量化
