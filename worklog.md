# Worklog

## 2026-06-02 Synthetic DFN 生成器更新

### 背景

这次改动的目标是让 `src/generate_synthetic_dfn.py` 更接近 Teng et al. (2025) 的合成 DFN 数据构造方式：在原有随机线段生成器基础上，加入分形位置分布、更可控的幂律长度分布，以及可设置平均方向的 von Mises 方向分布。

这次没有重写训练代码，也没有自动重生成 `data/synthetic_dfn_128`。现有数据集如果没有手动重跑生成脚本，仍然是旧生成器生成的数据。

### 改动以前的旧数据生成参数

旧版本命令行默认参数如下：

```text
--num_samples 10000
--image_size 128
--out_dir data/synthetic_dfn_128
--seed 42
--min_fractures 20
--max_fractures 80
--min_length None
--max_length None
--min_width 1
--max_width 2
--length_distribution lognormal
--orientation uniform
--von_mises_kappa 4.0
```

旧版本中，如果 `--min_length` 和 `--max_length` 不显式指定，会使用：

```text
min_length = image_size * 0.08
max_length = image_size * 0.65
```

对默认 `image_size = 128`，也就是：

```text
min_length = 10.24 px
max_length = 83.2 px
```

旧版本每张图的裂隙数量为：

```text
num_fractures ~ randint(20, 80)
```

旧版本的裂隙中心位置为均匀采样：

```text
center_x ~ Uniform(0, image_size - 1)
center_y ~ Uniform(0, image_size - 1)
```

旧版本默认长度分布是 lognormal：

```text
length ~ LogNormal(mean=log((min_length + max_length) / 3), sigma=0.55)
length clipped to [min_length, max_length]
```

旧版本可选 `--length_distribution power_law`，但实现是：

```text
raw = Pareto(a=2.5) + 1
length = min_length * raw
length clipped to [min_length, max_length]
```

这个旧幂律没有命令行参数控制指数，且通过 clip 截断，容易在 `max_length` 附近堆积。

旧版本默认方向分布是均匀方向：

```text
angle ~ Uniform(0, pi)
```

旧版本可选 `--orientation von_mises`，但均值固定在 `0` 弧度，只能通过 `--von_mises_kappa` 控制集中度：

```text
angle ~ VonMises(mu=0, kappa=von_mises_kappa) mod pi
```

旧版本每条裂隙用 `cv2.line` 绘制，最终二值化：

```text
image = (image > 0) * 255
```

旧版本 metadata 只记录：

```text
sample_id
image_size
num_fractures
fractures: center_x, center_y, length, angle, width
```

不会记录位置分布、长度分布、方向分布等全局生成参数。

### 本次新增的生成控制

新增位置分布参数：

```text
--position_distribution {uniform,fractal}
--fractal_dimension 2.0
--fractal_levels 6
--fixed_cascade_orientation
```

`uniform` 保持旧行为。`fractal` 使用乘法级联式象限采样来近似论文中的分形裂隙中心分布。`fractal_dimension` 对应论文中的 `Dc`，当前允许 `(0, 2]`。当 `Dc = 2` 时，四象限概率为均匀分布；当 `Dc < 2` 时，会产生一个主象限和三个较小概率象限，从而形成聚集。

新增可控幂律长度参数：

```text
--power_law_exponent 2.5
```

当 `--length_distribution power_law` 时，现在使用截断幂律的反 CDF 采样，而不是旧的 `Pareto + clip`。这样 `a` 可以显式控制，也更接近论文中以 `lmin`、`lmax` 和幂律指数描述裂隙长度的方式。

新增 von Mises 平均方向参数：

```text
--von_mises_mean_degrees 0.0
```

当 `--orientation von_mises` 时，现在可以同时设置平均方向和集中度：

```text
angle ~ VonMises(mu=von_mises_mean_degrees, kappa=von_mises_kappa) mod pi
```

metadata 现在会额外记录：

```text
position_distribution
fractal_dimension
fractal_levels
length_distribution
power_law_exponent
orientation
von_mises_mean_degrees
von_mises_kappa
```

### 运行环境要求

本 repo 的 Python 命令默认使用 `conda dfn` 环境运行。该环境应安装 `requirements.txt` 中列出的依赖，包括：

```text
opencv-python
tqdm
numpy
pillow
```

生成器继续强依赖 `cv2` 和 `tqdm`：使用 `cv2.line` / `cv2.imwrite` 绘制和保存图像，使用 `tqdm` 显示生成进度。不保留无依赖 fallback。

### 论文式数据生成示例

10 条裂隙、`64 x 64`、位置分形维数 `Dc=2`、长度幂律指数 `a=2`、方向 von Mises 平均 `30 deg`、集中度 `kappa=5`：

```bash
conda run -n dfn python src/generate_synthetic_dfn.py \
  --num_samples 30000 \
  --image_size 64 \
  --out_dir data/synthetic_dfn_teng_10 \
  --min_fractures 10 \
  --max_fractures 10 \
  --position_distribution fractal \
  --fractal_dimension 2.0 \
  --fractal_levels 6 \
  --length_distribution power_law \
  --power_law_exponent 2.0 \
  --orientation von_mises \
  --von_mises_mean_degrees 30 \
  --von_mises_kappa 5 \
  --min_length 5 \
  --max_length 20
```

20 条裂隙可以改为：

```bash
--min_fractures 20 --max_fractures 20
```

50 条裂隙按论文思路应提高到 `128 x 128`：

```bash
--image_size 128 --min_fractures 50 --max_fractures 50
```

### 验证记录

已完成的本地检查：

```text
conda run -n dfn python -m py_compile src/generate_synthetic_dfn.py
```

已用 `/tmp` 路径做两组 smoke test：

```text
/tmp/gendl_dfn_default_conda
/tmp/gendl_dfn_teng_conda
```

使用 `conda dfn` 环境时，默认参数能生成 PNG 和 JSON metadata；论文式参数也能生成 PNG 和 JSON metadata。该环境中已确认：

```text
cv2 4.13.0
tqdm 4.67.3
```

分形概率检查：

```text
Dc = 2.0 -> [0.25, 0.25, 0.25, 0.25], sum_sq = 0.25 = 2^-2
Dc = 1.5 -> [0.528684, 0.157105, 0.157105, 0.157105], sum_sq = 0.353553 = 2^-1.5
```

截断幂律采样检查示例：

```text
min_length = 5
max_length = 20
power_law_exponent = 2.0
sample min ~= 5.0007
sample max ~= 19.974
sample mean ~= 9.2335
```

### 当前注意事项

- 这次只是模仿论文的数据构造部分，不包含论文中的条件连通筛选、EDSR+PHT 线段级统计、EDFM/GEOS 正演或 DREAM(ZS) 反演。
- 当前分形采样是轻量近似，适合先复现实验控制项；如果后续要严格对齐论文，应进一步实现完整 MPP 流程和对应的 `Dc` 估计验证。
- 旧的 `data/synthetic_dfn_128` 不会自动更新。要让训练使用新分布，需要显式重跑生成脚本并把配置中的 `data.image_dir` 指向新输出目录。
