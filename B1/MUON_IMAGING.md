# 最基本的缪子成像示例

这个版本现在是论文 PoCA 基线的简化实现：

```text
4 GeV mu-  --->  上游两层  --->  U 型铝制被测物  --->  下游两层  --->
                (拟合入射径迹)                    (拟合出射径迹)
```

程序对每个缪子记录四个探测器上的位置，拟合入射和出射径迹，计算散射角，并用两条拟合直线的最近点中点得到 PoCA 散射位置。结果写入 `muon_tracks.csv`。

```text
event,upstream_x_mm,upstream_y_mm,...,downstream_x_mm,downstream_y_mm,...,dx_mm,dy_mm
```

其中 `poca_x_mm`、`poca_y_mm`、`poca_z_mm` 是 PoCA 散射位置，`theta_x_mrad`、`theta_y_mrad` 和 `theta_mrad` 是散射角分量和总散射角。

散射角的基本计算为：

```text
theta_x = atan2(outgoing_x, outgoing_z) - atan2(incoming_x, incoming_z)
theta_y = atan2(outgoing_y, outgoing_z) - atan2(incoming_y, incoming_z)
```

## 编译和运行

在项目目录执行：

```bash
cmake --build build -j2
cd build
./exampleB1 run1.mac       # 10 个缪子，适合快速检查
./exampleB1 run2.mac       # 1000 个缪子，获得更多统计量
./exampleB1 run3.mac       # 10000 个缪子，固定体素 PoCA 基线
```

输出文件会自动写入相邻的数据处理目录：
`/Users/xuyaoyang/geant4-test/muon_analysis/muon_tracks.csv`。
因此需要从项目的 `build` 目录运行程序。

## 留出 L 型目标验证（不覆盖 U 型基线）

`MUON_TARGET_SCENARIO=l_shape` 会在同一套四层探测器中放置一个
120 mm 外轮廓、30 mm 壁厚、100 mm 厚度的 L 型铝目标。
它没有参与前面 U 型场景的角度阈值、体素比例或事件质量规则选择。
先在 `B1` 目录生成独立 CSV，再由数据处理目录中的脚本分析：

```bash
cd /Users/xuyaoyang/geant4-test/B1
cmake --build build -j4
MUON_TARGET_SCENARIO=l_shape \
MUON_TRACK_OUTPUT=/Users/xuyaoyang/geant4-test/muon_analysis/muon_tracks_l_shape.csv \
./build/exampleB1 run_l_shape_validation.mac

cd /Users/xuyaoyang/geant4-test
MPLBACKEND=Agg muon_analysis/.venv/bin/python \
  muon_analysis/run_heldout_l_shape_validation.py
```

原始事件保存在 `muon_analysis/muon_tracks_l_shape.csv`，派生 CSV、图和
实验说明保存在 `muon_analysis/heldout_l_shape_results/`。现有的
`muon_analysis/analysis.ipynb` 末尾也有对应的阅读与展示单元格。
该验证仅改变目标形状，不能替代新材料、自然宇宙线或真实探测器验证。

## 双分离铝块：冻结规则的独立形状验证

本场景由两块 50 × 60 × 100 mm 的铝块组成，中间留 30 mm 空隙。
所有角度筛选、细化比例、局部收缩和回退阈值保持前期设置，不用
双铝块结果重新选择参数。运行命令：

```bash
cd /Users/xuyaoyang/geant4-test/B1
cmake --build build -j4
MUON_TARGET_SCENARIO=double_block \
MUON_TRACK_OUTPUT=/Users/xuyaoyang/geant4-test/muon_analysis/muon_tracks_double_block.csv \
./build/exampleB1 run_double_block_validation.mac

cd /Users/xuyaoyang/geant4-test/muon_analysis
MPLBACKEND=Agg ./.venv/bin/python run_double_block_validation.py
```

原始数据与分析结果分别保存在 `muon_tracks_double_block.csv` 和
`double_block_validation_results/`，不会覆盖 U/L 场景。结果图、配对
表和中文结论也可从同一个 `analysis.ipynb` 的末尾查看。这里的自动
单窗口定位误差仅作兼容指标：双目标应主要结合 IoU、两块最低召回率
和空隙伪影判断。

## T 型新场景：两半事件一致性细化的冻结规则验证

细化规则、随机种子与判定标准在运行新场景前已写入
`muon_analysis/CONFIDENCE_REFINEMENT_PROTOCOL.md`。脚本会核对规则源码
SHA-256，不允许误将改动后的版本称作同一次独立验证。

```bash
cd /Users/xuyaoyang/geant4-test/B1
cmake --build build -j4
MUON_TARGET_SCENARIO=t_shape \
MUON_TRACK_OUTPUT=/Users/xuyaoyang/geant4-test/muon_analysis/muon_tracks_t_shape.csv \
./build/exampleB1 run_t_shape_validation.mac

cd /Users/xuyaoyang/geant4-test/muon_analysis
MPLBACKEND=Agg ./.venv/bin/python run_t_shape_independent_validation.py
```

原始事件保存为 `muon_tracks_t_shape.csv`；全部配对结果、预注册判定、
图像与实验说明在 `t_shape_independent_results/`，同一个 `analysis.ipynb`
末尾也能查看。该预注册标准未通过，不能把新方法写成已验证的轮廓提升。

## 建议学习顺序

1. 修改 `DetectorConstruction.cc` 中 U 型铝制目标的材料、尺寸和位置。
2. 修改 `PrimaryGeneratorAction.cc` 中缪子的能量、方向和位置。
3. 在 `EventAction.cc` 中增加偏转角、散射半径等输出量。
4. 再把固定位置的单束缪子改成具有角度和位置分布的缪子源。
5. 最后使用大量事件和二维网格统计，制作真正的成像图。
