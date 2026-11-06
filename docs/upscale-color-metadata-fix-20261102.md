# AI 放大：灰度色彩配置与异常 EXIF 修复

2026-11-02。备份位于 `build/rollback/upscale-metadata-fix-20261102/`，包含修改前代码、应用、数据库、两张原图和海滩图已完成的 AI 结果。

## 确认的原因

- `街头雨天行人.jpg` 为 500 × 745 的 L 灰度 JPEG，嵌入 GRAY / Dot Gain 20% ICC。旧代码先转 RGB，再按 GRAY 输入色彩配置创建转换，触发 `cannot build transform`。错误发生在上传及提交 AI 任务之前。
- `海滩_木头_波浪_阴天.jpg` 已完成的 1600 × 1066 AI 候选没有损坏。源图 EXIF 的 ShutterSpeedValue 为 -32768/0，ExposureBiasValue 为 0/0。旧代码重新序列化 EXIF 时，Pillow 对未定义的负数分数推断为无符号类型，触发 `'L' format requires 0 <= number <= 4294967295`；失败发生在源文件替换之前。

`build/qa/upscale-metadata-fix-20261102/check_actual_files.py` 从备份归档加载旧版实现，分别复现两个相同错误，再用新版对相同图片副本验证修复。记录见 `actual-file-report.json`。

## 实现

- 临时 AI 输入按 ICC 对应的 L、RGB 或 CMYK 像素模式转换为 sRGB。保留透明通道，输入不携带源文件的拍摄元数据。
- 替换时将 sRGB 候选转回源色彩空间。GRAY JPEG 保持 L 灰度和原 ICC；PNG 支持灰度透明通道。WebP 只能编码 RGB，因此使用与像素一致的 sRGB 配置，避免将 GRAY ICC 错贴到 RGB 像素。
- 逐字段清理 EXIF 中分母为零、NaN、无穷大的数值，保留有效参数、相机信息、日期、GPS 和互操作目录。同步更新根目录及 EXIF 子目录中已有的宽高字段，方向归一为 1。
- 被省略的字段路径保存在替换记录的 `metadata_omissions` 中，页面显示省略数量。完整原始字节保留在备份中，“恢复原图”仍逐字节恢复原始文件及其元数据。
- 像素旋转使用仅包含方向信息的临时副本，避免 `exif_transpose` 意外重写无关的异常元数据。缩略图也使用此路径，确保恢复带旋转及异常 EXIF 的原图后不会在更新缩略图时再次报错。
- 沿用原有文件校验、原子替换、数据库恢复、标签和语义保留逻辑。已有 AI 候选继续可用。

色彩模式要求可参照 [Pillow ImageCms 文档](https://pillow.readthedocs.io/en/stable/reference/ImageCms.html)；本次行为以本机 Pillow 12.2.0 及实际图片的复现结果为准，未升级依赖。

## 验证记录

- 新回归覆盖 GRAY ICC、RGB ICC、透明度、JPEG/PNG/WebP、异常分数、有效负数分数、EXIF 子目录尺寸、方向 1–8、替换提示和原始字节恢复。
- 实际图片副本中，旧版错误均复现；新版灰度输入成功准备、海滩图已有候选成功替换，两者恢复后的 SHA-256 均与原始备份一致。灰度图此项的候选为测试缩放图，不代表已完成神经网络推理。
- 完整检查结果、桌面验收及安装检查保存在 `build/qa/upscale-metadata-fix-20261102/`。
- macOS 原生 Cocoa 下，放大、输入缩小、预览标记及本次回归共 108 项通过；解码、扫描和视频另 17 项通过；可靠性另 17 项通过，合计 142 项。先前 offscreen 合并运行出现过 Qt 控件构造 `SystemError`，原生 Cocoa 全量重跑及真实桌面操作均通过，原始日志保留。
- 从同一打包应用制作的独立验收应用，使用 `~/Library/Caches/EidoryQA/upscale-metadata-20261102/` 中的图库及图片副本。最初在工作目录启动时，诊断采样显示停在 SQLite 文件打开，迁至缓存目录后正常启动；未修改系统权限。
- 真实桌面点击“开始放大”，街头雨天行人图通过本机 SeedVR2 7B 完成 4 倍推理，得到 2000 × 2980 的候选；点击替换后保存为 L 灰度及原 GRAY ICC，再点击恢复，SHA-256 与原始备份一致。
- 海滩图沿用用户已生成的 AI 候选副本，桌面点击替换成功，显示跳过两项无效拍摄参数，再恢复后 SHA-256 与原始备份一致。正式图库的已有候选无需重新生成。
- 已打包安装至 `/Applications/Eidory.app` 并重新启动，严格签名验证通过，现有图标保持一致。正式图库仍为 28,397 张；数据库逐表对照通过，两张正式原图及海滩图已有 AI 候选的文件校验值均未改变。
