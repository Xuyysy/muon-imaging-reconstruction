# 运行、失败和来源核验

运行前登记提交10ef793；run-001实际数值源码提交82ae010，完整SHA见其config。
最终提交SHA由Git交付给出，不写入自身避免循环。全部改动仅在本study。

## 结果与历史保护

- 17项初始测试通过；审查追加3项后20/20通过。是本study的完整测试，
  不称旧公共全套测试通过；旧100k/120k MC未运行。
- run-001完成，所有数值积分交叉核查达1e-8；15个求解均认证，NLL跨度
  6.82e-13、最大独立KKT4.30e-7，未调整旧优化器容差。
- 1536个PCG64种子部件逐字节重放一致；全部6144条方法/重复记录重算一致；
  run manifest的18个文件SHA-256通过；原始基线跟踪路径没有修改。
- source snapshot保存运行时17项测试版本；当前20测试的差异被明确登记，
  不把后加测试回填为run-001时已有的证据。

来源哈希（完整明细见run-001/evidence.json和verification/archive-review-002.json）：

| 产物 | SHA-256 |
|---|---|
| run-001 config.json | 2ba60091b63e2a94ee2b01d41cc4a5ab1a0d0afb19837671198ec7a18c45f2a6 |
| run-001 evidence.json | a088251341b3276e136557b51668e0a4033b695f80d773cd0897f12e5f2094fa |
| mode_energies.npz | 9f31617fdb8030983f8c4493227094024ba24b215f7178cb6b7b9981eca877cf |

配置记录Python/NumPy/平台、完整命令、代码SHA、工作树状态、输入报告哈希、
固定动量/密度单位、scenes、counts和PCG64种子规则。共享源码只读调用。

## 保留的失败

1. tests-red-001：实现前缺diagnostics模块，预期RED，记录保留。
2. archive-command-001/archive-review-001：当前测试增加3项，首版审查器错误
   地将其当作运行源码被改。核验改为检查实际运行Git提交与不可变快照，
   只允许显式登记的新增测试差异；科学脚本与输入仍逐字节一致。
   archive-command-002/archive-review-002通过，旧失败不覆盖。
3. 前置环境探测的3个Python环境均缺SciPy，未安装依赖；改用NumPy及
   整数Gamma尾有限和、确定性积分。此为实施选择，不是放松数值容差。
4. 初次工作区写入受沙箱限制；使用已授权的独立worktree写入权限后运行。
   未切换或修改主工作区。

未出现被丢弃的科学重复、拟合失败或积分容差失败；弱信号漏检、误支持、
候选劣于经典上界等负面结果原样保留。阶段0的裁切失败也没有修改。

## CPU预算

每个实际数值/测试/复核命令由budget_run.py保存完整stdout/stderr、退出码、
子进程user+system CPU和wall。单线程环境、每进程RLIMIT_CPU；本轮核心
研究runner硬限300秒，测试/审查各120秒，所有记录累计仍远低于3600秒。
截至archive-command-002，累计可归属子进程CPU约5.959秒（约.00166核时）。
最终静态交付核查另以独立记录计入，精确总计在其JSON。

Git、shell、文件读取、解释器前置探测及预算wrapper自身开销没有逐项完整计量；
估计本轮本地辅助CPU为秒至几十秒量级，不把它报告为零。LLM推理、远程
文献/工具服务CPU不可观测，不伪称计入本地核时。没有运行任何长后台任务，
未接近1CPU核时上限。人日属于原规划，不声称本轮精确耗用2–3个人日。

## 复现命令（仓库根目录）

环境按config固定Python/NumPy版本；无SciPy/GPU依赖。应使用全新tag/run ID；
脚本拒绝已存在目录。下面命令仅供复现，本轮没有自动运行run-002。

```bash
python3 sci_method/studies/ml-nonneural-feasibility-001/budget_run.py \
  --tag tests-replay-001 --limit 120 -- python3 -m unittest discover \
  -s sci_method/studies/ml-nonneural-feasibility-001 -p 'test_*.py' -v

python3 sci_method/studies/ml-nonneural-feasibility-001/budget_run.py \
  --tag diagnostic-replay-002 --limit 300 -- python3 \
  sci_method/studies/ml-nonneural-feasibility-001/diagnostics.py --run-id run-002
```

`review_checks.py`有意固定读取canonical run-001作只读审查；不作为任意run的
通用API。后续复制或适配必须说明来源。最终未合并main、未更新已证论文台账。
