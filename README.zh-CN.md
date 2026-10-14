# ManiSoft-RL：软体操作与飞机加油末端对接

这是基于 Gymnasium 的研究代码包，包含软体连续臂抓取、连续体加油末端对接，以及独立的刚性伸缩加油杆仿真。两个方向共用环境接口、并行执行与策略晋级检查，因此放在同一个仓库中。

[English documentation](README.md) · [发布验证记录](PUBLICATION_VALIDATION.md)

## 包含什么

| 环境 | 观测与动作 | 运行依赖 |
| --- | --- | --- |
| `ManiSoft-COLL-v0` | 85 维状态；18 维软臂驱动加 1 维夹爪控制 | 外部 ManiSoft 及其任务资产 |
| `ManiSoft-AerialRefuel-v0` | 82 维状态，可选 RGB-D；18 维弯曲加 1 维伸缩 | 外部 ManiSoft、PyElastica、MuJoCo |
| `ManiSoft-AerialRigidBoom-v0` | 状态或 RGB-D；左右控制面与伸缩共 3 维动作 | MuJoCo，可独立运行 |

仓库还提供 SAC/TD3 训练入口、向量环境、回放脚本、环境合同哈希和父策略/候选策略的配对晋级检查。晋级要求保留父策略全部成功案例，并至少新增一个成功案例。

飞机任务的边界是进入预接触编队后的末端对接。整机飞行控制、尾流耦合、真实飞行数据标定和实机加油尚未在这里验证。RGB-D 回放也不等同于已训练的视觉策略或 VLA 系统。

## 安装与最小运行

使用 Python 3.11 或 3.12：

```bash
git clone https://github.com/hangzhou2321/-.git manisoftrl
cd manisoftrl
python -m venv .venv
```

Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`；Linux/macOS 激活：`source .venv/bin/activate`。

```bash
python -m pip install -e ".[test,rigid-aerial]"
python -m pytest
python examples/rigid_aerial_smoke.py --output artifacts/rigid_smoke --steps 20
```

最后一条命令会输出 RGB 图像、以米为单位的深度数组和 `report.json`。它使用固定动作验证仿真链路，不是训练后的成功演示。需要可用的 OpenGL 渲染环境。

软体臂和 COLL 抓取还需单独安装上游 [ManiSoft](https://github.com/buaa-colalab/ManiSoft)，并通过 `runtime_root`、`asset_root` 和 `case_dirs` 指定运行时、资产和案例目录。详见英文说明中的示例；本仓库不包含该外部运行时或数据集。

## 目录

- `src/manisoftrl/`：环境、执行接口与晋级检查。
- `examples/`：训练、回放、并行运行和最小验证。
- `tests/`：包级合同和回归测试。
- `demo/`：飞机可视化资产来源与坐标绑定说明。
- `AERIAL_REFUELING_VISION_CONTRACT.md`：空中对接视觉与控制合同。

## 公开版本边界

公开版以原有的可复用代码包为基础，核心源码与本地开发目录逐文件核对一致。桌面和 F 盘的历史训练结果、远端完整备份、实验草稿与中间检查点保留在原位置，未作为已验证研究结论一起发布。

本次只复核安装、33 项包级测试和刚性加油杆最小仿真；没有重新训练策略，也没有重跑软体臂完整物理实验。复现这些实验需要另行准备相应运行时、资产、检查点和冻结评估配置。

代码沿用已有 MIT 许可证。第三方运行时和飞机模型各自遵守原许可证，见 [NOTICE.md](NOTICE.md)。
