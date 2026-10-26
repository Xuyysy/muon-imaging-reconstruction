# 测试与未达预期记录

以下摘录来自本轮终端输出，不是新物理实验。

## 1. RED：先写测试，缺少核查模块

命令：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s sci_method/studies/ml-credibility-stage0-001 -p 'test_*.py' -v`

首次运行：`ModuleNotFoundError: No module named 'audit_checks'`；1个loader error。
这是实现尚未存在的测试开发记录，不是前向模型失败。

## 2. 解析实现后，11通过、1项预期失败

原测试：

```python
def test_edge_clipping_breaks_null_not_a_claim_of_power(self):
    self.assertGreater(audit.analyze()['tilted_clipped']['relative_covariance_change'], .001)
```

两次相同输入均得到：

```text
AssertionError: 6.84674866339067e-05 not greater than 0.001
Ran 12 tests
FAILED (failures=1)
```

独立矩积分及原积分实现一致；根因不是舍入/积分bug，而是最初错误地期望材料核变化足以令**含测量噪声总协方差**超过0.1%。材料协方差行和相对变化为27/676≈3.994%，总协方差仅0.00684675%。

保留CHECK_SPEC原门槛0.001，在checks.json记录`initial_clipping_threshold_met=false`，run状态为`completed_with_failed_effect_expectation`。没有降低门槛、改变源/误差/λ尺度或挑选新几何以获得正结果。该门槛只是本次诊断的事前效应预期，并非SCI正式成功标准。

最终单元测试修正为验证：门槛确实未达且被记录、总变化大于原舍入预算、材料变化匹配独立解析27/676。测试绿色只证明工具忠实记录负结果，不将这项科学预期改成成功。原断言和失败值在此保留，可用同一analyze结果直接复现。

实现检查还将旧NLL接口调用保持为原单位：原kernel(mm³)×λ(mm^-1)，不把缩放后的kernel和u误标成旧API的单位；这不改变其数值协方差或已有算法。

尝试用tee保存首次失败时遇到worktree写权限限制（`Operation not permitted`），没有生成声称完整的原始log；因此这里明确标为终端摘录。最终测试/正式确定性run的日志通过授权执行保存，既有结果未动。
