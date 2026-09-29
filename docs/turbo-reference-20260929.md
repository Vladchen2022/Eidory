# 生成参考 · Turbo 默认工作流

## 使用

在项目节点进入「找参考 → 生成参考」，顶部直接选择工作流。

- **Viggle Turbo · 6 步**：升级后默认，Qwen 2.1 官方 BF16 基座 + Viggle Turbo v0.2.1 r256 BF16 权重；强度 1、Euler、专用 6 步调度、BasicGuider（无额外 CFG），默认 1024 × 1024。
- **千问原版 · 40 步**：保留原来的模型与采样配置。现有用户的 `comfy.template` 原样保留，不覆盖；本机旧工作流默认尺寸仍为 1152 × 2048、Euler/simple、CFG 1。
- 选择会记住。切换工作流保留图片描述，恢复对应工作流默认尺寸。Turbo 尺寸要求为 32 的倍数。
- 自定义 API 导入是第三个独立选项，不覆盖 Turbo 或原工作流。
- 「再生成」沿用选中结果保存的完整工作流，在选择器明确显示「原结果参数」，不随当前默认值改变。
- 缺少 Turbo 节点或权重时明确报错，不擅自切到耗时更长的原版。AI 放大仍使用原有 SeedVR2 流程。

工作流按照本机 `05_千问2.1_ViggleTurbo_6步_BF16.json` 核对；使用 `RandomNoise.noise_seed`，不向高级采样器错误写入 `seed`。原 ComfyUI 文件不被改写，输出仍采用临时传输与 Eidory 候选保存机制。

## 验证

最终相关回归测试 **61 项通过，0 项失败，6.29 秒**，日志在 `build/qa/turbo-reference/final-feature-tests.log`。覆盖生成参考及 AI 放大原有行为，以及默认升级、旧配置保留、自定义导入隔离、Turbo 采样连接与种子、尺寸检查、缺失模型、历史快照再生成。屏幕测试发现对话框默认按钮可能导致按 Enter 意外生成，已取消这些按钮的隐式默认行为，并加入回归验证。

屏幕测试使用独立数据库 `/tmp/eidory-turbo-qa-20260929/data`，项目「双工作流验收｜江南茶馆」，不在真实用户项目上生成测试记录。比较使用同一段文字和 1024 × 1024 尺寸，保留每套工作流本来的采样参数。实际生成、预览与保存结果见本次对比页面及测量记录。

对比页面与原图：`build/qa/turbo-reference/report/`。原始计时证据：`build/qa/turbo-reference/measurements.json`；屏幕证据：`build/qa/turbo-reference/screens/`。每套仅一次成功样本，随机种子不同，缓存及运行顺序影响耗时，不推断稳定加速倍数或整体画质排名。

| 实测 | Turbo 6 步 | 原版 40 步 |
| --- | ---: | ---: |
| ComfyUI 执行耗时 | 42.976 秒 | 232.552 秒 |
| 创建任务至候选可用 | 43.405 秒 | 233.483 秒 |
| 生成、预览、采用、看板 | 通过 | 通过 |

本次总等待时间比为 **5.38**。设备为 M4 Max、128 GiB 统一内存，缓存保留；诊断任务和主动取消的任务不计入结果，也可能影响缓存。两张图已永久保存到 `~/Pictures/Eidory Imports/AI生成/双工作流验收｜江南茶馆/`，独立测试数据库备份在 `build/qa/turbo-reference/test.sqlite3`。

已更新 `/Applications/Eidory.app`，签名校验通过，安装版启动并首次打开生成参考时默认显示 Turbo。与安装前一致性备份比较，38 张业务表中 36 张逐行一致；测试入口所在项目及节点仅 `updated_at` 改变，内容一致。图片记录 25,130 张，标签、语义索引、参考关联及旧 `comfy.template` 均保持一致，数据库 `quick_check` 为 `ok`。核对记录在 `build/qa/turbo-reference/install-audit.json`。

## 回退

基线提交 `e537dfa0396637e9215a6c8aa242f4f884f7d931`；标签 `checkpoint/before-turbo-reference-20260929`；实施分支 `codex/turbo-reference-20260929`。

`build/rollback/turbo-reference-20260929/` 保存升级前 `Eidory.app`、SQLite 一致性备份与 `repository.bundle`。优先只回退程序：退出应用，另存当前版本，再用备份应用恢复 `/Applications/Eidory.app`。旧工作流设置仍在，旧程序可继续使用。不要强制重置有其他改动的工作目录，不要覆盖运行中的数据库。
