# 五篇博客对应的独立运行代码

每个 `01`～`05` Python 文件都包含自己的环境、网络、PPO、GAE、训练、评估和绘图逻辑，**没有 `cartpole_pg` 或其他项目内部导入**。把任意一个脚本复制到其他目录，安装三个依赖后即可从头训练，不需要先运行前篇，不会自动寻找原项目的模型。

## 文件与博客一一对应

| 可独立运行的文件 | 对应博客 | 相比前篇保留 / 新增 |
|---|---|---|
| [01_discrete_ppo.py](01_discrete_ppo.py) | [01：平衡基线](../01_用PPO建立倒立摆平衡基线.md) | 完整动力学、离散 Actor–Critic、PPO + GAE |
| [02_continuous_ppo.py](02_continuous_ppo.py) | [02：连续力](../02_只改动作空间让推力连续.md) | 保留 PPO/GAE；高斯 + tanh，奖励仍为存活 |
| [03_smooth_reward_ppo.py](03_smooth_reward_ppo.py) | [03：平滑奖励](../03_在连续控制中加入平滑奖励.md) | 保留连续力；加入倾角、位置、用力与力变化代价 |
| [04_position_ppo.py](04_position_ppo.py) | [04：目标位置停留](../04_继承平滑控制实现目标位置停留.md) | 保留连续力与控制代价；增加位置误差、速度目标和 30 秒评估 |
| [05_robust_ppo.py](05_robust_ppo.py) | [05：随机扰动恢复](../05_继承位置控制学习随机扰动恢复.md) | 完整保留位置奖励；随机脉冲训练、批量抗扰评估、外力箭头动画 |

代码有意重复基础组件，使每个文件本身就是完整的教学实现。它们不是依赖原项目包的启动壳。离散与连续模型不兼容，加载时会检查参数名、形状和有限值。


生成的 HTML 页面、图表和 GIF 统一使用英文标签及 ASCII 符号（例如 `angle`、`deg`、`+/-`），不依赖中文字体。文本输出显式使用 UTF-8，适配不同系统的默认编码。修改脚本后需重新运行，已有输出文件不会自动更新。

## 代码注释

注释集中解释动力学、PPO/GAE、连续动作雅可比、梯度与评估边界等关键实现；简单赋值、循环及页面模板不再逐行重复说明。生成页面仍使用英文。注释精简前后已核对 Python 抽象语法树，确认执行逻辑和字符串内容一致，见 [检查记录](annotation_verification.json)。

## 安装与直接运行

在此文件夹打开终端：

```bash
python3 -m pip install -r requirements.txt

python3 01_discrete_ppo.py
python3 02_continuous_ppo.py
python3 03_smooth_reward_ppo.py
python3 04_position_ppo.py
python3 05_robust_ppo.py
```

上述五条命令彼此独立。默认第一篇训练 200,704 步，其余 401,408 步，每批 2048 步向上取整；最终评估默认 30 回合。代码在导入 NumPy 前设置单线程默认值，并使用无窗口绘图后端，不需要显示器、Gymnasium 或 PyTorch。

**独立从头运行与复现博客中的微调链是两种模式。** 第一至三篇的原实验本来就是从头训练；第四、五篇的原实验依赖前一模型微调。如果不传 `--initial`，第四、五篇也能运行，但有限预算下的表现可能与博客不同。脚本会把实际模式、初始模型路径及训练预算写入 `metrics.json`，不会把从头结果标成微调结果。

## 不等训练，直接看已发布模型

`checkpoints` 中包含五个原实验的种子 0 验证最佳模型，`provenance.json` 记录来源和 SHA-256。以下命令只评估和生成图像动画：

```bash
python3 01_discrete_ppo.py --evaluate checkpoints/01_discrete.npz
python3 02_continuous_ppo.py --evaluate checkpoints/02_continuous.npz
python3 03_smooth_reward_ppo.py --evaluate checkpoints/03_smooth_reward.npz
python3 04_position_ppo.py --evaluate checkpoints/04_position.npz
python3 05_robust_ppo.py --evaluate checkpoints/05_robust.npz
```

`--evaluate` 不进行训练。第四、五篇在没有 `--initial` 时，对照明确标为“未训练基线”，不是自动加载旧位置模型。历史博客中的旧策略对照仍见原结果数据。

## 按文章的实际继承方式继续训练

```bash
# 第四篇：从第三篇已发布的平滑模型开始微调
python3 04_position_ppo.py --initial checkpoints/03_smooth_reward.npz --output outputs/position_finetune

# 第五篇：从第四篇已发布的位置模型开始微调
python3 05_robust_ppo.py --initial checkpoints/04_position.npz --output outputs/robust_finetune
```

如果要把自己刚训练的结果逐步传给下一篇：

```bash
python3 03_smooth_reward_ppo.py --output outputs/my_smooth
python3 04_position_ppo.py --initial outputs/my_smooth/policy.npz --output outputs/my_position
python3 05_robust_ppo.py --initial outputs/my_position/policy.npz --output outputs/my_robust
```

第二、三篇在原实验中是同架构、同预算的独立奖励对照，不需要先运行第二篇才能运行第三篇。可以显式指定兼容的连续模型进行新微调实验，但不能把这种新实验当作博客原始结果。

## 常用参数

```bash
# 快速检查整条流程，不代表模型已经学好
python3 03_smooth_reward_ppo.py --steps 2048 --episodes 2 --output outputs/smoke

# 更换随机种子或训练预算
python3 04_position_ppo.py --seed 1 --steps 802816 --initial checkpoints/03_smooth_reward.npz --output outputs/position_seed1

# 指定固定目标，目标范围 [-1.5,1.5] m；需要重新检查效果
python3 04_position_ppo.py --target 0.5 --initial checkpoints/03_smooth_reward.npz --output outputs/target_05

# 自选外力测试强度，不改变训练中的 1～3 N 随机脉冲课程
python3 05_robust_ppo.py --evaluate checkpoints/05_robust.npz --forces 0 3 -3 10 -10 15 -15 --output outputs/strong_test

python3 05_robust_ppo.py --help
```

`--initial` 与 `--evaluate` 互斥；前者加载参数继续训练，Adam 动量重置，后者跳过训练。`--episodes` 控制最终独立评估回合数，不改变训练中的验证集大小。第五篇目前只支持目标 0 m；`--forces` 仅控制最终评估的外力强度，动画另外保留 +3 N 演示。

## 输出在哪里

默认路径相对于**脚本所在目录**，不依赖从哪里启动：

```text
code/outputs/01_discrete_ppo/
code/outputs/02_continuous_ppo/
code/outputs/03_smooth_reward_ppo/
code/outputs/04_position_ppo/
code/outputs/05_robust_ppo/
```

第一至四篇输出 `demo.html`、`ppo.gif`、训练曲线、状态/推力曲线、指标及轨迹。训练模式输出 `policy.npz` 与 `policy_final.npz`；仅评估模式不复制或覆盖输入模型。第四篇额外输出逐回合位置指标与全部轨迹。

第五篇输出批量扰动指标、逐回合数据，动画在 `animation/demo.html` 与 `animation/pulse.gif`。外力不参与执行器 ±10 N 限幅，上一动作观测仍只记录控制器自己的推力。动画在 ±12° 倾角或轨道越界时停止，不继续画倒地。

输出目录若已有同名文件会覆盖；不同试验请指定不同 `--output`。这些脚本不覆盖原项目 `results`。

## 与原实验的关系及验证

这五个文件复制并内嵌了原实现的核心计算，没有改动 PPO、奖励、GAE 掩码或动力学参数。新增内容主要是独立 CLI、显式可选初始模型、检查点兼容性检查、输出路径隔离和各篇自带报告。

独立性验证把每个文件分别复制到 `/tmp` 的单独目录，使用 Python 隔离模式且不提供任何模型，运行 2048 步训练、独立评估并生成 PNG/GIF/HTML；这只是运行链路检查。加载附带原模型可查看博客展示的控制效果。完整训练结果仍有随机性，短训练不保证成功。

验证记录见 [verification.json](verification.json)。独立复制或解压代码文件夹后可直接运行；说明中的博客链接需要保留完整项目目录才可跳转，不影响脚本执行。
