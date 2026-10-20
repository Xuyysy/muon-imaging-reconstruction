# 固定动量散射前向模型与局部信息原型

状态：数值/数学可行性原型；不代表已完成重建算法或 SCI 创新验证。所有本轮观测均由手工指定径迹和合成高斯分布生成，没有运行 Geant4，也没有使用真值 mask、目标标签或 oracle Top-k。

## 1. 变量、单位与范围

- 四层名义探测器中心：\(z=(-400,-300,300,400)\,\mathrm{mm}\)。这里用的是名义中心作为简化几何，并非 CSV 中每个 hit 的薄闪烁体表面 z。
- 每平面 hit：\((x,y)\) 的独立高斯测量误差，标准差 \(\sigma_h\) mm；z 位置视为已知几何，不加误差。
- 固定缪子动量 \(p=4\,\mathrm{GeV}/c\)，\(\beta\simeq1\)。这是理想固定动量近似，**不外推到自然宇宙线动量谱**。
- 区域参数 \(\lambda_v=1/X_{0,v}\)，单位 \(\mathrm{mm}^{-1}\)，要求 \(\lambda_v\geq0\)。本原型不含材料能损、吸收、空气/支撑结构、非高斯散射尾、磁场或径迹间相互作用。
- 残差单位 mm；角度以无量纲弧度计。Fisher 信息的数值依赖参数单位、局部参考点及路径设计，不能脱离这些定义解释成通用阈值。

## 2. hit 到下游残差的线性变换

将四层 hit 按 plane-major 顺序展成
\[
h=(x_{u1},y_{u1},z_{u1},x_{u2},y_{u2},z_{u2},x_{d1},y_{d1},z_{d1},x_{d2},y_{d2},z_{d2})^T.
\]
只取 hit x/y；四个观测量按
\[
y=(r_{x,d1},r_{x,d2},r_{y,d1},r_{y,d2})^T,
\quad r_{a,d}=a_d-[(1-t_d)a_{u1}+t_da_{u2}],
\quad t_d={z_d-z_{u1}\over z_{u2}-z_{u1}}.
\]
定义 \(J\in\mathbb R^{4\times12}\)，则 \(y=Jh\)。z 列全为零，因为 plane z 已知。对名义中心，本原型的 \(t_{d1}=7,t_{d2}=8\)，所以 x 的两行分别含上游系数 \((6,-7)\)、\((7,-8)\)，下游 hit 系数为 1；y 行相同。

这里的下游 residual 是针对**上游拟合直线**的观测，不是 PoCA，也不表示真实散射发生于某个单点。

## 3. 完整探测器残差协方差

每层 x/y 测量误差相互独立且方差为 \(\sigma_h^2\)：
\[
\Sigma_h=\operatorname{diag}(\sigma_h^2,\sigma_h^2,0,\ldots,\sigma_h^2,\sigma_h^2,0),\qquad
\Sigma_{\rm det}=J\Sigma_hJ^T.
\]
由于两个下游残差共享同一对上游 hit，协方差不对角。每一个投影的 \(2\times2\) 块是
\[
\sigma_h^2\begin{pmatrix}86&98\\98&114\end{pmatrix},
\]
而 x-y 交叉块在独立各向同性 hit 误差假设下为零。例如 \(\sigma_h=0.3\) mm 时，每个块为
\(\begin{psmallmatrix}7.74&8.82\\8.82&10.26\end{psmallmatrix}\,\mathrm{mm}^2\)。解析矩阵对称半正定；非零 \(\sigma_h\) 时为正定。单元测试用固定 seed 的 120,000 次高斯 hit 扰动核对解析非零项，最大相对误差门槛预先设为 2%；x-y 解析零项另外按模型结构精确核对，不用有限样本交叉协方差冒充精确零。

## 4. Fermi–Eyges 路径积分协方差

对一条名义直线路径 \(x(z)=x_0+s_xz,\ y(z)=y_0+s_yz\)，以及下游平面 \(j,k\)，小角散射造成的残差协方差为
\[
K_{jk}[\lambda]=\kappa^2\int_{\mathrm{path}}(z_j-z)_+(z_k-z)_+\lambda(\mathbf r(z))\,ds,
\qquad \kappa={13.6\,\mathrm{MeV}\over \beta pc}.
\]
固定 \(p=4\,\mathrm{GeV}/c,\beta=1\) 时 \(\kappa=0.0136/4\) rad。用固定网格分段常数密度：
\[
K[\lambda]=\sum_v\lambda_vD_v,\quad
(D_v)_{jk}=\kappa^2\int_{\mathrm{path}\cap v}(z_j-z)_+(z_k-z)_+\,ds.
\]
\((z_j-z)_+(z_k-z)_+\) 单位 mm²，\(ds\) 单位 mm，因此 \(D_v\) 单位 mm³；乘 \(\lambda_v\,[\mathrm{mm}^{-1}]\) 得 mm²。正部杠杆臂保证因果性：散射发生于某个测量面之后时，对该面位置残差的贡献严格为零。实现对直线路径与轴对齐盒体求精确区间交，再解析积分关于 z 的二次多项式，并用 \(ds/dz=\sqrt{1+s_x^2+s_y^2}\)。

对 x/y 两投影假设独立、相同的 Brownian angular kicks，因此四维残差散射矩阵是两个相同的 2×2 下游块，x-y 块为零；同一投影两个 downstream residual 的协方差一般非零。它是路径积分的统计模型，不是给每事件指派“真实散射点”。

另外用 seed `20261020`、100,000 条径迹和 1 mm 中点 kick 离散，对 `lambda=10^-4 mm^-1` 的解析路径协方差做独立 Monte Carlo 核对。测试在运行前固定 3% 非零矩阵项相对误差容差；实测最大相对差为 `0.238%`，且下游残差交叉协方差是非零并参与比较。该检验验证了离散 Brownian-kick 近似下的积分实现，不验证 Geant4 的散射物理模型。

因果范围回归测试还验证：完全位于第一个下游测量面之后的 voxel 对该面方差和跨面协方差为零，但可影响更下游面；目标位于两探测器组之间时，对两个下游面仍有非零方差及交叉协方差。因此此修正不改变 run-007 所采用目标区间的有效响应。

## 5. 高斯似然、梯度与 Fisher 信息

事件 i 的总协方差为
\[
C_i(\lambda)=\Sigma_{{\rm det},i}+\sum_v\lambda_vD_{iv}.
\]
在事件独立、残差零均值的原型假设下，省略常数项的负对数似然为
\[
\mathcal L(\lambda)=\frac12\sum_i\{\log|C_i|+y_i^TC_i^{-1}y_i\}.
\]
解析梯度：
\[
{\partial\mathcal L\over\partial\lambda_v}=
\frac12\sum_i\left[\operatorname{tr}(C_i^{-1}D_{iv})-
y_i^TC_i^{-1}D_{iv}C_i^{-1}y_i\right].
\]
实现不加隐式 diagonal ridge；\(C_i\) 非正定即明确报错。仅测量噪声为零、材料散射为正时，测试模型仍可为 SPD；两者同时为零则为退化 delta 极限，普通连续高斯 NLL 不存在，代码拒绝计算。

协方差参数的期望数据 Fisher 为
\[
F_{ab}=\frac12\sum_i\operatorname{tr}(C_i^{-1}D_{ia}C_i^{-1}D_{ib}).
\]
本研究中的“局部可辨识”严格指在预定参考点附近，似然对参数或参数**对比方向**的曲率/信息；它不是全局唯一恢复保证。对局部参数 \(\theta\) 与 nuisance \(\eta\)，将 nuisance profile 掉后的局部信息取 Schur 补
\[
F_{\theta\mid\eta}=F_{\theta\theta}-F_{\theta\eta}F_{\eta\eta}^{+}F_{\eta\theta}.
\]
报告 data Fisher、nuisance-profiled contrast 信息和先验/正则精度是三件事。加入 \(\Lambda_{\rm prior}\) 后的 \(F+\Lambda_{\rm prior}\) 仅表示局部后验精度示意，**不能说先验让观测新增了信息**，也不能由后验方差变小证明数据足以辨识。

## 6. 预先定义的合成可辨识性试验

每个场景固定 200 条手工路径、两个 hit 误差条件 \(\sigma_h=0,0.3\) mm、固定 4 GeV/c 和相同的局部参考密度 \(10^{-4}\,\mathrm{mm}^{-1}\)。比较一个粗区、两个等分区、四个等分区。Fisher 数值秩阈值固定为最大特征值的 \(10^{-10}\)；没有扫描门槛或挑选最优图像。

| 场景 | 覆盖设计 | 结果（data Fisher rank，粗/2/4 参数） | 解释 |
|---|---|---|---|
| 横向四条带、均匀覆盖 | 200 条垂直径迹均匀分到四条带；4 细区每区 50 条，2 区每区 100 条 | \(\sigma_h=0.3\) mm：1/1、2/2、4/4；profile 全局 common-scatter nuisance 后，局部信息秩为 0、1、3 | 该 nuisance 严格等于所有局部散射核之和，是共同目标散射方向的代数退化负例，不解释为真实空场背景。由 4 区模型可辨识的是 3 个局部对比，不是 4 个独立的绝对密度。 |
| 横向覆盖缺失（理论失败例） | 全部 200 条径迹只穿过左侧第一条带 | 1/1、1/2、1/4；2/4 参数模型出现精确零信息方向；profile common nuisance 后局部秩均为 0 | 未穿越区域没有似然信息，增加参数只会产生不可辨识自由度；总事件数增加若仍不改变路径覆盖也不能修复。 |
| 四段深度切分 | 200 条同一横向位置的垂直径迹均穿过四段 z 区间 | 0.3 mm 时 raw rank：1、2、3（4 参数模型最小特征值为数值近零）；profile nuisance 后秩：0、1、2 | 两段深度可提供一个深度对比；四段参数只有约三种独立纵向二阶矩响应，至少一组合方向无可用信息。测量误差进一步压低弱方向。 |

横向均匀覆盖的四区 raw Fisher 四个特征值均为 \(5.1229\times10^4\)（参数单位由 \(\lambda\) 定义决定）；剖除上述“所有局部核之和”代数 nuisance 后为一个数值零模及三个 \(5.1229\times10^4\) 模式。\(\sigma_h=0\) 时相应每个 raw 特征值为 \(10^{10}\)。同一固定路径/参数化下，0.3 mm hit error 将此信息缩小约 \(1.952\times10^5\) 倍；这是理想手工例中的模型敏感性，不是现实探测器预测。

这里必须区分三种背景处理：已知背景以固定协方差加到似然中；未知背景只有在独立空场数据进入联合似然后才可估计；若背景导数与目标对比导数相同，则目标对比不可辨识。第三种是代数退化的负控制，不表示实际环境背景。

四段深度切分在 \(\sigma_h=0.3\) mm 时 raw Fisher 特征值约为
\((0,1.289,642.4,51228.9)\)，因此即使 200 条径迹都穿过四段，最弱非零局部组合仍比强方向弱约 4 万倍；在 \(10^{-10}\) 相对秩阈值下为 rank 3。profile common-scatter nuisance 后仅两个模式高于同一相对阈值。这里的近零模式与 nuisance-profile 的方向不同，不能混成一种不确定性。

数据似然与先验的单元测试用 data Fisher `diag(2,0)`：其 rank 为 1；仅给先验精度 `diag(0,5)` 后的总精度 rank 变为 2，但原 data Fisher 保持逐元素不变且 rank 仍为 1。这恰好说明正则化可稳定/约束参数，不等于观测获得新信息。

## 7. 科学解释边界与阶段判断

1. 四层配置每投影给出两项下游残差，协方差结构包含三个纵向积分矩（两个方差与一个交叉协方差）；x/y 在理想各向同性假设下提供重复统计信息，而不是六个可任意独立指定的深度响应。跨事件的横向路径覆盖可增加空间区分能力，但本原型的深度例显示四段细分已出现近零方向。
2. 可辨识区域取决于路径覆盖；缺少命中覆盖时 rank 从满秩直接降为 1。总 events 数本身不足以保证局部细节可辨识。
3. hit 误差会以完整相关协方差进入 \(C_i\)，并压低 Fisher 信息；若把四个 residual 当独立标量，会错误估计信息与似然。
4. 这些结果给 ICAR-LR 的窄化版本提供**数学动机**：任何 split 决策都应看似然对新增对比参数的信息，并把 nuisance/profile、先验影响分开。但它们不证明该方法优于现有方法，也不证明已达到有意义的真实成像分辨率。
5. 下一阶段可进入 split/merge 原型设计，但在进入真实 Geant4/历史事件前，应先由 ChatGPT/研究者审查本报告，并冻结参数缩放、信息判据、目标/背景 nuisance 形式、分裂几何与负例。仅可先做小型数学原型；没有必要现在扫描图像质量门槛。

## 8. 复现与结果

- 模块：[`src/sci_method/forward_model.py`](src/sci_method/forward_model.py)、[`src/sci_method/identifiability.py`](src/sci_method/identifiability.py)。
- 测试：[`tests/`](tests/)；hit-error covariance MC 容差 2%；散射 kick covariance MC 容差 3%（seed `20261020`、100,000 tracks、1 mm bins）。
- 试验脚本：[`studies/run_forward_identifiability.py`](studies/run_forward_identifiability.py)。
- 完整输出及哈希：[`studies/forward-prototype-001/results/run-007/`](studies/forward-prototype-001/results/run-007/)，包括 `config.json`、手工路径、合成残差、信息矩阵 CSV/JSON、`evidence.json`、完整源快照及哈希。`run-001` 至 `run-006` 保留为早期/范围不同或来源快照不完整的诊断记录，不用于主要结论，不覆盖。
- 复现时需选新的 `--run-id`；脚本对已存在路径失败退出，避免覆盖证据。
