# numpy-neural-network-image-classifier

基于 `numpy` 实现的二分类神经网络，用于识别游戏「Logres of Swords and Sorcery」中的目标怪物。项目包含数据集制作、训练、推理与可视化学习曲线的完整流程，用来记录我的学习与实践过程。

This is a binary image classifier implemented using `numpy`. It detects whether a target monster appears in screenshots from the game “Logres of Swords and Sorcery”. The project includes dataset preparation, training, inference, and visualization of learning curves to document my personal learning journey through practical implementation.

---

## 简介 | Overview

- 使用 `numpy` 手写前向传播、反向传播、参数更新等核心逻辑；不依赖高阶深度学习框架。
- 利用 `Pillow` 处理图像，`matplotlib` 展示学习曲线与性能指标。
- 提供数据集制作脚本，支持将整张截图切片为固定网格的小图并统一尺寸。
- 训练后可进行单张图片预测与文件夹批量评估（Precision/Recall）。

- Implements forward/backward propagation and parameter updates purely with `numpy`.
- Uses `Pillow` for image I/O and `matplotlib` for plots of training dynamics.
- Includes a dataset preparation script to tile screenshots into fixed-size crops and resize them.
- Supports single-image inference and batch evaluation with Precision/Recall.

---

## 目录结构 | Project Structure

```
numpy-neural-network-image-classifier/
├── main.py                  # 训练、可视化与推理入口 | Training, plots, inference
├── screenshotsToImages.py   # 数据集切片与缩放 | Screenshot tiling and resizing
├── label_gui.py             # 可视化标注工具（勾选→复制并加前缀）| GUI labeling tool
├── label_classification.py  # 统计并按前缀分类到子目录 | Count and split by prefix
├── labelImage/              # 训练数据目录（文件名前缀决定标签）| Training images (filename prefix → label)
├── label_0/                 # 评估用负类图片 | Evaluation negatives
├── label_1/                 # 评估用正类图片 | Evaluation positives
├── similarityImage/         # 供 GUI 选择的候选图片（可选）| Candidate images for GUI (optional)
├── Screenshots/             # 原始截图 | Raw screenshots
├── LICENSE                  # MIT 许可 | MIT License
└── README.md                # 项目文档 | This file
```

---

## 环境依赖 | Requirements

- `Python >= 3.8`
- `numpy`
- `Pillow`
- `matplotlib`

安装依赖 | Install dependencies:

```
pip install numpy pillow matplotlib
```

---

## 数据集制作 | Dataset Preparation

1) 将原始截图放入 `Screenshots/`。

2) 运行切片与缩放脚本：

```
python screenshotsToImages.py
```

- 脚本会按 `grid_width` × `grid_height` 网格大小在整张截图上滑动切片，并保存到 `subImage_{grid_width}_{grid_height}/`；随后将其统一缩放到 `new_width` × `new_height`，保存到 `sizeImage_{grid_width}_{grid_height}/`。
- 请确保截图的尺寸与网格大小匹配，避免切片越界（例如截图宽高是 `grid_width`/`grid_height` 的整数倍）。

3) 人工标注：将最终用于训练的图片放到 `labelImage/`，并通过文件名前缀决定标签：

- 以 `1_` 开头 → 正类（目标怪物存在）
- 非 `1_` 开头 → 默认负类（目标怪物不存在），可以使用 `0_` 前缀以示区分

4) 评估集（可选）：将样本分别放入 `label_1/`（正类）与 `label_0/`（负类），用于批量评估 Precision/Recall。

---

## 图像标注 | Image Labeling

方式 A：直接改文件名前缀 | Method A: rename by prefix

- 将图片放入 `labelImage/`，以 `1_`（正类）或 `0_`（负类）作为文件名前缀。

方式 B：使用 GUI 标注工具 | Method B: use the GUI tool

1) 准备候选图片：将需要筛选的图片放入 `similarityImage/`（或你希望标注的任意目录）。

2) 运行标注工具：

```
python label_gui.py
```

- 如需更改来源/目标目录与前缀，在 `label_gui.py` 的 `main()` 中修改：

```
source_folder = 'similarityImage'  # 候选图片目录
dest_folder = 'labelImage'         # 复制目标目录
prefix = '0'                       # 复制时添加到文件名前的前缀（例如 '1' 或 '0'）
```

- 用法：
  - 界面中滚动浏览缩略图，勾选需要复制的图片。
  - 点击 `Label` 按钮，所选图片将被复制到 `dest_folder`，并加上设定的 `prefix` 作为文件名前缀。
  - 为两个类别分别执行一次（正类设 `prefix='1'`，负类设 `prefix='0'`）。
  - 支持懒加载，滚动到底部会继续加载后续图片（`load_step` 默认为 30）。

3) 可选分类与统计：

```
python label_classification.py
```

- 在 `label_classification.py` 的 `main()` 中保持或修改：

```
source_folder = 'labelImage'
prefix_0 = '0'
prefix_1 = '1'
```

- 脚本会统计 `labelImage/` 内以 `0_` 与 `1_` 开头的文件数量，并复制到 `labelImage/label_0` 与 `labelImage/label_1` 子目录。
- 如果你希望与 `main.py` 的默认评估路径（项目根目录的 `label_0/` 与 `label_1/`）一致，可选择以下任一方案：
  - 方案 1：将 `labelImage/label_0` 与 `labelImage/label_1` 复制或移动到项目根目录。
  - 方案 2：在 `main.py` 中把评估路径改为 `test_folder_path = 'labelImage/label_1'` 与 `test_folder_path = 'labelImage/label_0'`。

---

## 快速开始 | Quick Start

1) 在 `main.py` 中确认或调整以下关键参数：

- `folder_path = 'labelImage'` 训练数据目录
- `num_iterations = 1800` 训练迭代次数
- `learning_rate = 0.1` 学习率
- `train_ratio = 0.7` 训练/验证划分比例
- `learning_rates = [0.001, 0.01, 0.1, 1]` 用于绘制准确率-学习率曲线
- `m_values = list(range(2, 180, 10))` 用于绘制代价-样本量曲线
- `test_image_path` 单张图片测试路径

2) 启动训练与可视化：

```
python main.py
```

- 训练过程中会打印周期性的训练/验证代价，并绘制三类学习曲线：
  - Accuracy vs. Learning Rate（准确率-学习率）
  - Cost vs. Number of Iterations（代价-迭代次数）
  - Cost vs. m（代价-训练样本量）

3) 单张图片预测：在 `main.py` 设置 `test_image_path` 指向目标图片，程序会输出预测类别（`0` 或 `1`）。

4) 批量评估：程序会对 `label_1/` 与 `label_0/` 进行整体预测统计，打印 Precision 与 Recall：

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

---

## 实现细节 | Implementation Details

- 网络结构：单隐层前馈网络（输入 → Sigmoid(隐藏层) → Sigmoid(输出层)），输出为二分类概率。
- 损失函数：二分类交叉熵（支持可选 L2 正则项函数 `compute_cost_lambda`）。
- 训练流程：
  - `initialize_parameters` 随机初始化权重与偏置
  - `forward_propagation` 前向传播计算激活
  - `compute_cost` 计算损失
  - `backward_propagation` 反向传播计算梯度
  - `update_parameters` 按学习率更新参数
- 数据处理：图片统一缩放到 `28×28`，flatten 后归一化到 `[0, 1]`。

- Network: one-hidden-layer MLP with Sigmoid activations, outputting binary probabilities.
- Loss: binary cross-entropy (optional L2 via `compute_cost_lambda`).
- Training loop: initialize → forward → cost → backward → update.
- Images are resized to `28×28`, flattened, and normalized to `[0, 1]`.

提示 | Notes:

- 当前未固定随机种子，重复运行可能得到不同结果；如需复现实验可在开头添加 `np.random.seed(42)`。
- 标注规则基于文件名前缀，任何非 `1_` 前缀的文件都会被当作负类。

---

## 学习经过 | Learning Journey

- 起点：希望从数学与实现层面理解二分类神经网络，于是基于 `numpy` 实现了一个最小可用的多层感知机（MLP）。
- 数据：从游戏截图出发，使用脚本按网格切片并统一尺寸；用文件名前缀进行简易标注（`1_` 为正类，其余作负类）。
- 模型：单隐层、`sigmoid` 激活与二分类交叉熵；隐藏层大小选为 `100` 以平衡表达能力与训练稳定性。
- 训练：实现前向/反向传播与参数更新；通过三类曲线观察超参数影响（学习率、迭代次数、样本量）。
- 评估：统计单图预测与批量 Precision/Recall，并据此微调学习率与迭代轮次。
- 坑点：随机种子未固定导致结果波动；学习率过大易震荡、过小收敛慢；切片尺寸与截图分辨率需严格匹配。

- Motivation: understand binary classifiers from math and implementation, so I implemented a minimal MLP with `numpy`.
- Data: start from game screenshots; tile and resize; simple filename-based labels (`1_` positive, otherwise negative).
- Model: one hidden layer, `sigmoid` activations, binary cross-entropy; hidden size `100` to balance capacity and stability.
- Training: implement forward/backward and updates; visualize learning rate, iteration count, and dataset size effects.
- Evaluation: single-image predictions and batch Precision/Recall to guide hyperparameter tuning.
- Pitfalls: no fixed random seed → variance; too-large LR oscillates, too-small LR is slow; crop size must match screenshot resolution.

---

## 常见问题 | FAQ

- 切片结果过多或越界？请检查 `screenshotsToImages.py` 中的网格大小与截图分辨率是否匹配；建议截图宽高为 `grid_width` 与 `grid_height` 的整数倍。
- 学习率选择？可参考准确率-学习率曲线，通常从 `1e-3 ~ 1e-1` 之间尝试。
- 训练不收敛或过拟合？尝试调整 `num_iterations`、`train_ratio`、`hidden_size` 或加入正则项（使用 `compute_cost_lambda` 逻辑）。

---

## 贡献 | Contributing

这是我的学习练习项目，欢迎提出 Issue 或 PR 交流与建议；如有更直观的可视化或练习题，也欢迎分享。

This is a personal learning project. Issues and PRs are welcome for discussion and suggestions; feel free to share visualizations or exercises.

---

## 许可 | License

本项目使用 MIT 许可协议。详见 `LICENSE` 文件。

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 致谢 | Acknowledgements

这是一次个人学习的记录与分享，如有不妥之处欢迎指正与交流。

This repo documents my own learning and understanding; happy to receive feedback and discuss.
