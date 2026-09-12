# 原生标题栏随主题更新

原因：主窗口此前固定使用 macOS Dark Aqua 外观，而且 `_macos_titlebar_applied` 让后续切换主题直接跳过原生标题栏设置。Qt 样式表没有覆盖 NSWindow 的背景。

修复：浅色与米白使用 Aqua，三个深色主题使用 Dark Aqua；原生窗口背景取当前主题的 `base` 色，与工作台顶部一致。主题变化时更新颜色，窗口结构只在首次显示时配置，避免主题切换反复调整窗口样式。

原生接口依据：[Apple 标题栏背景说明](https://developer.apple.com/documentation/appkit/nswindow/titlebarappearstransparent?language=objc)、[Apple 外观说明](https://developer.apple.com/documentation/AppKit/NSAppearance)。PySide6 保持 6.9.3。

## 验证

- 主题与工作台专项测试 27 项通过，包括主题切换刷新、反复显示不重复配置、侧栏恢复和筛选／选图／看板状态保留。
- 独立测试安装包实际查看全部五种主题，标题栏已显示对应背景，深浅文字正常。深色切回浅色、再切回米白也正确。
- 米白主题下放大窗口再恢复大小，标题栏与左右侧栏正常；退出后重新启动仍显示米白标题栏。
- 构建日志：`build/qa/titlebar-build.log`；测试日志：`build/qa/titlebar-tests.log`。

## 回退

- 修复前提交：`0e88e1b`。
- 修复前标签：`checkpoint/titlebar-20260912`。
- 分支：`codex/theme-titlebar`。
- 旧应用备份：`build/rollback/Eidory-before-titlebar-20260912.app`。

本次不修改图库数据库结构或原图。

已更新 `/Applications/Eidory.app`，保留用户选择的米白主题。替换下来的应用另存为 `build/rollback/Eidory-installed-before-titlebar.app`。
