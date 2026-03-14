# Goodhart Cascade — 代码完善设计

**日期**: 2026-03-14
**目标**: 将现有实现与终版 Proposal 完全对齐，修复正确性问题

## 变更清单

### 高优先级（正确性）

1. **Granger 因果差分**: `analysis/temporal.py` — 在 Granger 检验前对序列做一阶差分
2. **Type B 三层判定**: `eval/temptation.py` — 增加 if-elif 链检测层和代码极短检测层
3. **Logprob 鲁棒性**: `eval/calibration.py` — 扫描多个 token 位置寻找 Yes/No
4. **Sandbox 内存限制**: `utils/code_exec.py` — 用 resource.setrlimit 实施内存限制

### 中优先级（功能完备性）

5. **LLM 诱惑生成**: `data/generate_temptation.py` — 增加 OpenAI API 生成选项（Type A/C）
6. **退化窗口法**: `analysis/quality_submetrics.py` — 用 3 点连续窗口确认退化
7. **Cognitive complexity**: `eval/code_quality.py` — 嵌套深度加权计算
8. **Type A 语义冲突**: 与 #5 合并，LLM 生成合理但错误的输出

### 低优先级（清理）

9. **死代码清理**: `analysis/temporal.py` — 移除未用的 n_bkps 参数
10. **全面测试验证**: 确保所有 188+ 测试通过
