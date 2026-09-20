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

## 建议学习顺序

1. 修改 `DetectorConstruction.cc` 中 U 型铝制目标的材料、尺寸和位置。
2. 修改 `PrimaryGeneratorAction.cc` 中缪子的能量、方向和位置。
3. 在 `EventAction.cc` 中增加偏转角、散射半径等输出量。
4. 再把固定位置的单束缪子改成具有角度和位置分布的缪子源。
5. 最后使用大量事件和二维网格统计，制作真正的成像图。
