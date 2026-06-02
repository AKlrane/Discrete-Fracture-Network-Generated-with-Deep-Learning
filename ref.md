# Teng et al. (2025) DFN-GAN 方法笔记

## 文献信息

- 论文：Zheng Teng, Hui Wu, Jize Zhang, Xin Ju, Shengwen Qi. *Generating high-fidelity discrete fracture networks from low-dimensional latent spaces using generative adversarial network*.
- 期刊：*International Journal of Rock Mechanics and Mining Sciences*, 196, 106301, 2025.
- DOI：`10.1016/j.ijrmms.2025.106301`
- Zotero key：`ST75R6ZA`
- 本地 PDF：`/Users/cranekoh/zotero-local/Teng et al. - 2025 - Generating high-fidelity discrete fracture networks from low-dimensional latent spaces using generat.pdf`

这篇文章的核心目标是把二维离散裂隙网络（Discrete Fracture Network, DFN）从高维、变长、难反演的显式几何参数空间，压缩到低维潜变量空间。作者用标记点过程（Marked Point Processes, MPP）生成满足地质统计先验的训练样本，再训练 Wasserstein 生成对抗网络带梯度惩罚（WGAN-GP），使一个低维向量 `z` 能确定性地产生一张 DFN 图像。

## 核心问题

传统 DFN 参数化通常需要直接描述每条裂隙的位置、长度、方向和端点。这样有三个问题：

- 维度高：二维场景中一条裂隙至少需要多个几何参数，裂隙数增多后参数空间迅速变大。
- 维度不固定：真实裂隙数量通常未知，反演过程中裂隙数也可能变化。
- 数据稀疏：钻孔、测井、示踪、压力和地球物理数据通常只提供局部约束，直接反演高维 DFN 会病态且计算昂贵。

文章的思路不是让 MPP 本身做反演参数化，因为 MPP 对同一组统计参数会随机生成很多不同 DFN，缺少“参数点 -> 唯一 DFN”的确定性映射。WGAN-GP 训练好以后，`G(z)` 则提供固定维度、确定性的映射：给定一个 latent vector，就生成一个对应 DFN。

## 整体流程

论文方法可以概括为四步：

1. 用 MPP 生成大量二维 DFN 训练图像。裂隙中心、长度、方向分别服从预设统计分布。
2. 用这些图像训练 WGAN-GP。生成器 `G` 学习从低维 `z` 到 DFN 图像的映射，critic `D` 学习区分训练 DFN 和生成 DFN。
3. 训练后从 latent space 随机采样，生成新 DFN。
4. 用增强深度超分辨率网络（Enhanced Deep Super-Resolution, EDSR）和概率霍夫变换（Probabilistic Hough Transform, PHT）提取裂隙线段，再统计裂隙数量、位置、长度、方向和连通性。

文章不只验证普通随机生成，还验证了三类更接近应用的问题：

- 能否保留训练样本的统计先验。
- 能否生成满足已知裂隙存在性和连通性的条件 DFN。
- 能否在真实露头图像和压力反演案例中作为低维参数化器使用。

## 训练样本构造

文章考虑二维 DFN，裂隙被表示为线段。MPP 的作用是先生成符合地质先验的一批训练样本，供 WGAN-GP 学习。

### 裂隙位置

裂隙中心位置用分形分布描述。文章使用相关维数 `Dc` 表征中心点空间聚集程度：

```text
Dc = lim_{r -> 0} log C2(r) / log r
```

其中 `C2(r)` 表示距离小于等于 `r` 的点对数量比例。`Dc` 越接近 2，二维平面内分布越均匀；`Dc` 较低时，裂隙中心更聚集。论文用乘法级联过程生成分形点分布，主要测试 `Dc = 1.5` 和 `Dc = 2`。

### 裂隙长度

裂隙长度服从幂律分布，常写作：

```text
n(l) = beta * l^(-a)
```

其中 `l` 是长度，`a` 是幂律指数，`beta` 是比例系数。`a` 越大，短裂隙相对越多。论文中常用 `a = 2` 和 `a = 3`，并用 `lmin`、`lmax` 作为长度硬边界。典型设置是：

```text
lmin = 5 m
lmax = 20 m
```

也做了敏感性测试，例如把 `lmin` 降到 `2 m`，或把 `lmax` 增到 `30 m`。

### 裂隙方向

裂隙方向服从 von Mises 分布：

```text
f(theta) = exp(kappa * cos(theta - mu)) / (2*pi*I0(kappa))
```

其中 `mu` 是平均方向，`kappa` 是集中参数。`kappa` 越大，方向越集中。论文测试了不同平均方向和集中度，例如：

```text
mu = 17 deg 或 30 deg
kappa = 5 或 20
```

合成训练样本的物理域尺寸通常是 `20 x 20 m2`，图像分辨率为 `64 x 64` 或 `128 x 128`。作者刻意使用较低分辨率来降低 WGAN-GP 训练成本；当裂隙数从 10/20 增加到 50 时，分辨率提升到 `128 x 128` 以保持线段可辨识。

## WGAN-GP 模型

论文采用 WGAN-GP，而不是普通 GAN 的 BCE 损失。普通 GAN 用 `D(x)` 表示真假概率，WGAN-GP 中的 `D` 更准确地说是 critic，输出一个实数分数。

生成器：

```text
G: z -> generated DFN image
```

critic：

```text
D: DFN image -> scalar score
```

critic 的训练目标可以写成：

```text
L_D = E[D(fake)] - E[D(real)] + lambda * E[(||grad D(x_hat)||_2 - 1)^2]
```

生成器的训练目标是：

```text
L_G = -E[D(fake)]
```

其中 `x_hat` 是真实图像和生成图像之间的随机插值，梯度惩罚用于满足 WGAN 所需的 Lipschitz 约束。论文采用 Gulrajani et al. 的常用设置：

```text
lambda = 10
learning rate = 1e-4
batch size = 64
```

网络结构上，论文使用残差网络（ResNets）：生成器包含多个上采样 residual block，critic 包含多个下采样 residual block。作者强调这是在模型容量和训练稳定性之间调参得到的结构。

## 验证方法

训练后，作者不只看生成图像是否“像裂隙”，还把生成 DFN 中的每条裂隙提取出来，比较统计分布是否和训练样本一致。

因为 `64 x 64` 图像分辨率较低，直接用 PHT 提取线段容易失败。论文先用 EDSR 做超分辨率，把原始 `64 x 64` DFN 放大到约 `300 x 300`，再用 PHT 提取线段。随后统计：

- 裂隙数量。
- 裂隙中心位置的分形维数 `Dc`。
- 裂隙长度分布和幂律指数 `a`。
- 裂隙方向分布及 von Mises 参数 `mu`、`kappa`。
- 条件生成场景下的已知裂隙连通率、交点数量、流动路径长度。

这个验证流程是文章和一般图像生成 baseline 的重要区别：作者试图证明生成器不仅学到了视觉纹理，还继承了 DFN 的地质统计先验。

## 主要实验

### 不同裂隙数量

论文准备三组主要合成数据集：

| 场景 | 每组训练样本 | 图像分辨率 | 训练迭代 |
| --- | ---: | --- | ---: |
| 10 条裂隙 | 30,000 | `64 x 64` | 100,000 |
| 20 条裂隙 | 30,000 | `64 x 64` | 100,000 |
| 50 条裂隙 | 30,000 | `128 x 128` | 100,000 |

`100,000` iterations 大约对应 200 epochs。论文报告在 NVIDIA GeForce RTX 4090 上，10/20 条裂隙场景约需 30 小时，50 条裂隙的 `128 x 128` 场景约需 140 小时。

### 不同 latent dimensionality

作者测试的 latent dimension 包括：

```text
2, 4, 8, 16, 32, 64
```

主要结论：

- latent dimension 增大时，裂隙的清晰度、直线性和视觉质量通常改善。
- 统计分布不一定随 latent dimension 增大显著改善；较低维度已经能保留主要统计特征。
- 对 20 条裂隙的例子，`LD = 8` 已能生成质量较高的 DFN。相比传统端点参数化约需 `20 x 4 = 80` 个参数，8 维 latent space 体现了很强的降维。
- 每个 latent 参数影响整体裂隙网络形态，而不是只控制单条裂隙；这使 DFN 随 latent vector 平滑变化，对后续反演有利。

### 统计保真性

论文对 10、20、50 条裂隙场景分别比较生成样本和训练样本。生成样本总体能接近训练样本的裂隙数量、分形位置分布、幂律长度分布和 von Mises 方向分布。

需要注意的是，统计误差不全来自生成模型。论文指出 PHT 检测本身会引入偏差：相近且方向类似的裂隙可能被合并，边界或交叉处的模糊区域可能被拆成多条短裂隙，靠近边界的裂隙会被截断。

## 条件 DFN 生成

文章进一步考虑“已知裂隙存在”和“已知裂隙连通”的约束。这类先验可能来自钻孔、井壁成像、水力试验或示踪试验。

实现方式不是改 WGAN-GP 损失，而是在训练样本层面施加约束：

1. 先随机生成大量 DFN。
2. 在每张 DFN 中强制放入若干预先存在的裂隙。
3. 指定其中一条为注入裂隙，其余为监测裂隙。
4. 检查注入裂隙和每条监测裂隙是否连通。
5. 只保留满足连通约束的样本。
6. 移除没有连接到任何预设裂隙的孤立裂隙。

论文测试 2 到 4 条预设裂隙。每个条件案例仍生成 `30,000` 个训练样本，训练后再生成 `12,800` 个样本分析。结论是：

- 生成样本能稳定恢复训练中强制存在的预设裂隙。
- 两条预设裂隙的条件案例中，连通成功率接近 90%。
- 当预设裂隙增加到 3 到 4 条，连通成功率下降；论文认为主要原因之一是裂隙检测误差，生成模型也有一定残余失败。
- `LD = 4`、`8`、`16` 在连通统计上差别不大，说明很低维 latent space 已能表达部分条件连通结构。

这个部分的关键思想是：WGAN-GP 通过训练样本继承先验。只要训练集只包含满足某类硬约束的 DFN，生成器就倾向于生成满足同类约束的 DFN，但无法保证 100% 满足。

## 真实露头与反演案例

### 真实露头

论文还用真实二维裂隙图像训练 WGAN-GP，包括英国 limestone outcrop 和冰川 crevasse 图像。做法是：

- 从约 `700 x 500` 的露头图像中，用 `64 x 64` 小窗口随机裁剪。
- 每个露头案例裁剪 `30,000` 个训练样本。
- 用 WGAN-GP 训练后随机生成新的 DFN patch。

视觉上，生成器能学到 limestone 中多组主方向、长裂隙和短连接裂隙，也能学到 glacier crevasse 中更弯曲、方向更集中的裂隙样式。不过作者没有对这些真实露头结果做同等精度的统计验证，因为稠密和弯曲裂隙会让 PHT 线段检测不可靠。

### 压力反演案例

反演案例把 WGAN-GP 当作低维 DFN 参数化器：

- 参考 DFN 含 8 条裂隙和注入/生产井。
- 两条穿过注入井和生产井的裂隙作为已知预设裂隙。
- 训练集包含 `30,000` 个满足这两条预设裂隙连通的 DFN。
- WGAN-GP 学习从 16 维 latent space 生成条件 DFN。
- 用 Embedded Discrete Fracture Method（EDFM）和 GEOS 做稳态流动正演。
- 在 49 个观测点提取压力数据。
- 用 DREAM(ZS) 多链 MCMC 在 latent space 中采样，最小化模拟压力和观测压力的偏差。

结果显示后验裂隙分布逐步向参考 DFN 的主要连通路径收敛，压力预测也能贴近参考值。文章认为原因有两点：一是固定维度 latent space 仍允许生成不同裂隙数量的 DFN；二是训练集连通约束减少了大量无效、断开的候选模型。

## 局限

论文明确提到几个限制：

- PHT + EDSR 不是完美的裂隙检测方案。相近线段、交叉处、模糊边界和边界截断都会影响统计估计。
- 对稠密、弯曲、真实露头裂隙，PHT 难以准确提取每条裂隙，因此真实露头部分主要依赖视觉判断。
- 分辨率提高会显著增加训练成本。`128 x 128` 的 50 裂隙案例需要约 23 GB 显存，训练时间超过 140 小时。
- 论文主要研究二维 DFN。扩展到三维 DFN 需要 3D 生成网络、更大训练集、更高计算成本，以及可靠的 3D 裂隙检测方法。
- 条件连通不是严格保证。即使训练集施加约束，生成样本仍可能有少量不连通情况；用于流动或示踪反演时需要筛选或额外约束。

## 和本仓库的关系

本仓库当前已经具备一个简化的 DFN 生成 baseline，但和 Teng et al. 的完整方法仍有明显差距。

### 相同点

- 本仓库也使用 WGAN-GP 作为 DFN 图像生成器，训练损失形式和论文一致：

```text
critic_loss = fake_score.mean() - real_score.mean() + lambda_gp * gradient_penalty
generator_loss = -fake_score.mean()
```

- 当前 `configs/wgan_gp_128.yaml` 中也使用：

```text
batch_size = 64
lr = 1e-4
lambda_gp = 10
num_epochs = 200
```

- 当前生成器和 critic 也是卷积网络，生成器从 latent vector 采样并输出 `128 x 128` 单通道 DFN 图像。
- 当前评估脚本也包含和 DFN 形态相关的指标，例如裂隙像素比例、连通分量、骨架长度、端点/交点数量、Hough line count 和方向直方图。

### 关键差距

- 数据生成：当前 `src/generate_synthetic_dfn.py` 使用均匀位置采样，长度可选 lognormal 或近似 power-law，方向可选 uniform 或 von Mises；没有实现 MPP 的分形位置分布，也没有按论文的 `Dc` 控制裂隙中心聚集程度。
- 条件约束：当前数据生成没有“已知裂隙存在性”和“注入-监测裂隙连通性”筛选，也没有移除与预设裂隙不连通的孤立裂隙。
- 模型结构：当前 `src/models/wgan_gp.py` 是 DCGAN 风格的 ConvTranspose/Conv 网络；论文使用 ResNet 上采样/下采样 block。
- latent 维度：当前默认 `latent_dim = 128`；论文强调测试极低维 latent space，如 `2/4/8/16/32/64`，并用低维性服务于反演。
- 验证方法：当前 `src/evaluation/evaluate_dfn.py` 用图像级指标和 Hough 方向直方图；论文使用 EDSR + PHT 提取线段，并进一步拟合 `Dc`、幂律长度参数 `a`、von Mises 参数 `mu/kappa`。
- 应用验证：当前仓库没有 EDFM/GEOS 流动正演，也没有 DREAM(ZS) 或其他 MCMC 在 latent space 中做压力/示踪反演。
- 真实露头：当前 README 明确项目不包括真实 outcrop 数据处理；论文则把 limestone 和 glacier outcrop 作为重要验证场景。

### 对本项目的启发

如果后续希望让本仓库更接近 Teng et al. 的论文方法，优先级可以是：

1. 在合成数据生成器中加入分形位置分布和更可控的幂律长度/von Mises 方向参数。
2. 增加低维 latent WGAN-GP 实验，例如 `latent_dim = 8/16/32`，观察生成质量和统计指标变化。
3. 增加线段级评估：提取裂隙中心、长度、方向，再拟合 `Dc`、`a`、`mu/kappa`。
4. 构造带预设裂隙和连通筛选的条件训练集，用于测试生成器是否能继承硬约束。
5. 在真正做流动反演前，先加入连通性和 flow-path length 这类无需完整物理正演的中间指标。
