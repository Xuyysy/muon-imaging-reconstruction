# SCI 新算法研究工作区

本目录服务于 SCI 新算法研究。当前定位为**第二阶段：最小数学前向模型与局部可辨识性原型**。ICAR-LR 仍是待验证候选，不是已成立的创新；本轮没有运行新的 Geant4 物理仿真，也没有开展图像质量或大规模性能实验。

## 当前研究策略

- SCI 方法论文是当前首要目标。硕士学位论文暂时只积累资料和证据，不在本阶段正式撰写。
- 保留 Geant4、PoCA、旧实验代码、输入 CSV、论文工作稿与阴性结果原位不动。旧脚本不是稳定 API；新代码不得把旧结果文件当作自己的输出位置。
- RC-AR 仅为历史候选名称，不是必须坚持的方案。已有研究已经覆盖大量 PoCA 自适应细分尝试；本轮把研究重心转向**原始四层击中、散射统计似然与可辨识性驱动的分辨率控制**。
- “原始击中统计重建”本身已有 2026 年近邻工作，因此不能单独作为创新点。当前主候选是：在物理散射似然和显式探测器测量协方差下，构造**可辨识性校准的自适应空间分辨率**，并输出哪些区域/尺度可由现有数据支持的置信信息。
- 该主候选的新颖性尚未确立。它必须与 MLEM/MLS-EM、ASR、PoCA、近期似然/原始击中方法及多尺度神经场方法公平比较；若理论和文献审查不能显示明确差异，应更换研究问题，而不是更换名字。

## 目录

- [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md)：代码与数据审查、文献比较、可辨识性分析、主/备选方案与风险。
- [`PROTOTYPE_PROTOCOL_DRAFT.md`](PROTOTYPE_PROTOCOL_DRAFT.md)：尚待研究者审查和冻结的最小原型实验草案。
- [`RESEARCH_EVIDENCE_INDEX.md`](RESEARCH_EVIDENCE_INDEX.md)：已有共享数据、基线、误差实验与旧研究的导航及证据边界。
- [`FORWARD_MODEL.md`](FORWARD_MODEL.md)：残差变换、完整测量协方差、Fermi–Eyges 路径积分、高斯似然/梯度、Fisher 信息定义和本轮合成诊断结果。
- [`SCI_PAPER_OUTLINE.md`](SCI_PAPER_OUTLINE.md)：SCI 论文初步章节结构与当前证据边界，不预写未验证结论。
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)：本阶段的实施边界与验收清单。
- [`src/sci_method/data_contract.py`](src/sci_method/data_contract.py)：只保留观测侧四层击中的只读输入适配契约；不含重建算法。
- [`src/sci_method/forward_model.py`](src/sci_method/forward_model.py)：固定 4 GeV/c、小角度、高斯误差的数值前向模型，不是完整 Geant4 替代品。
- [`src/sci_method/identifiability.py`](src/sci_method/identifiability.py)：数据似然 Fisher、nuisance profile 和先验精度分离工具。
- [`studies/forward-prototype-001/results/run-007/`](studies/forward-prototype-001/results/run-007/)：本轮 canonical 合成结果、固定配置、输入表、完整源代码快照/哈希和证据文件；旧 run-001 至 run-006 均保留。
- [`tests/`](tests/)：数据契约、协方差、梯度、信息矩阵与防覆盖测试。

## 当前状态与边界

状态：`phase-2 prototype complete; review required`。

1. `muon_analysis/muon_tracks.csv` 为既有开发资料；精确 Geant4 随机流和生成命令未核实，不能视为独立新测试。
2. 真值、目标掩膜和材料标签只允许在独立评价阶段载入；`sci_method/src/` 不接受这些字段。
3. 现有五次相关事件子抽样不是五次独立物理重复。
4. 任何新增实验均需在看结果前冻结研究问题、比较方法、主指标、成功标准、事件/随机流计划、分组和排除规则。
5. 结果是手工设定路径与合成高斯观测的局部信息诊断；不能外推为旧 CSV、真实探测器或自然缪子束的成像性能。
6. data Fisher 和 nuisance-profiled contrast information 与先验/正则精度分开报告。较小 posterior variance 不作为观测可辨识证据。
7. 脚本仅在新的 `run-*` 目录写结果；目标目录已存在时会失败，不会覆盖。复现需给出新的 `--run-id`。
8. `run-001` 至 `run-006` 是范围迭代期间的探索记录，保留但不作为 canonical 可复现证据（早期版本未保存完整源码快照）；本轮正式引用以 `run-007` 为准，它保存脚本及计算模块快照和哈希，并标记生成前工作区状态。

## 当前复现检查

在仓库根目录运行：

```bash
python3 -m unittest discover -s sci_method/tests -v
```

这验证输入契约和数学模块，不读取真值、不重建图像，也不改变任何历史数据。合成信息诊断可运行：

```bash
python3 sci_method/studies/run_forward_identifiability.py --run-id run-008
```

运行会生成新的合成结果；已有 `run-*` 目录不会被覆盖。运行前无需也不要启动 Geant4。
