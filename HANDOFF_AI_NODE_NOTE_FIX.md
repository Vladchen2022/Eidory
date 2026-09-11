# Eidory AI 节点说明保存问题交接

## Objective

继续修复 Eidory 右侧 AI 面板「创作节点」里的节点说明编辑/AI 补全功能。

用户要求的硬标准：
- 用户在任意节点「当前节点说明」里手动写的内容，切换到其他节点再切回，必须保留。
- 「当前节点信息补全」和「补全所有节点信息」必须把用户已写内容作为强约束扩展，不得清空、无视或生成矛盾内容。
- 必须用真实可见软件界面测试，不能只用后台/单元测试声称通过。

## Current State

仓库：`/Users/victorcloux/Desktop/Eidory`

当前有未提交修改。最新 `git status --short` 显示：

```text
 M src/eidory/config.py
 M src/eidory/core/llm_provider.py
 M src/eidory/ui/accessibility.py
 M src/eidory/ui/main_window.py
 M src/eidory/ui/project_board.py
 M tests/test_inspiration.py
 M tests/test_main_window_context_menu.py
 M tests/test_project_board.py
```

不要随手 revert。部分文件包含之前任务的未提交改动。

## Already Changed In This Session

`src/eidory/ui/main_window.py`
- 已加入 `QEvent` import。
- 已新增：
  - `_install_creative_node_editor_commit_guards`
  - `eventFilter`
  - `_creative_node_editor_should_commit_for_event`
- 已在 `_connect_signals()` 中把节点说明编辑器提交守卫装上。
- 守卫逻辑：在以下事件发生前调用 `_save_current_creative_node_editor_snapshot()`：
  - 节点说明输入框/搜索词输入框 `FocusOut`
  - 创作节点树、创作项目列表、左侧项目列表、右侧 tab bar `MouseButtonPress`
  - 主窗口 `WindowDeactivate` / `Close`
- 目的：把保存动作前移到用户点击切换节点之前，不再只依赖 `currentItemChanged` 后置保存。

`tests/test_main_window_context_menu.py`
- 已新增 `from PySide6.QtTest import QTest`
- 已新增测试：
  - `test_creative_node_tree_mouse_click_commits_visible_editor_text`
- 这个测试禁用自动保存后，在可视 widget 中写入「脏辫，右脸烧伤，蓝色义眼」，再用 `QTest.mouseClick` 点击另一个节点，验证旧节点写入 SQLite 且切回后文本还在。

`src/eidory/ui/accessibility.py`
- 之前为了让 computer-use 读 UI 树，加入了 `EIDORY_ENABLE_QT_ACCESSIBILITY=1` 旁路。
- 事实：全量打开 Qt Accessibility 会导致 PySide/Qt 在 macOS `NSAccessibility` 层崩溃。
- 下一轮要判断是否保留、收窄或移除这段旁路。

## Tests Already Run

通过：

```bash
.venv/bin/python -m pytest tests/test_main_window_context_menu.py -k "creative_node_note_is_saved_before_switching_nodes or creative_node_switch_captures_editor_text_even_if_autosave_was_missed or creative_node_tree_mouse_click_commits_visible_editor_text" -q
# 3 passed

.venv/bin/python -m pytest tests/test_main_window_context_menu.py -k "creative_node" -q
# 36 passed
```

没有在最新补丁后跑完整 `pytest -q`。

## App Packaging State

本轮曾执行：

```bash
./scripts/build_macos_app.sh
rm -rf /Applications/Eidory.app
cp -R /Users/victorcloux/Desktop/Eidory/dist/Eidory.app /Applications/Eidory.app
```

但这是在最后一次事件守卫补丁之前打包的。

所以现在：
- `/Applications/Eidory.app` 已被替换过，但相对当前源码是旧的。
- 如果要测最新补丁，必须重新打包并再次替换 `/Applications/Eidory.app`。

## Computer-use / Accessibility Findings

直接以 App bundle 启动并加环境变量：

```bash
EIDORY_ENABLE_QT_ACCESSIBILITY=1 /Applications/Eidory.app/Contents/MacOS/Eidory
```

computer-use 一开始能读到系统崩溃恢复弹窗。点 `Reopen` 后进入主窗口，但再次读取 UI 树触发崩溃。

最新 crash report 例子：

```text
/Users/victorcloux/Library/Logs/DiagnosticReports/Eidory-2026-07-09-173445.ips
```

关键点：
- `EXC_BAD_ACCESS`
- faulting stack 在 `NSAccessibility` / `AXCopyHierarchy`
- 这解释了为什么项目之前主动禁用/隐藏 Qt Accessibility：全量 AX 树扫描不稳定。

结论：下一轮不要简单全量开启 Qt Accessibility 来测试。可选策略：
- 用 `cliclick` + `screencapture` 做可见界面测试；
- 或实现更窄的测试专用 UI 暴露；
- 或做一个真实窗口内的自动化测试 harness，保证截图可见。

## Visible UI Test Attempt

为避免污染真实图库，创建了临时测试库：

```bash
EIDORY_DATA_DIR=/tmp/eidory-ui-node-retention-test
```

用脚本创建过测试项目：

```text
project_id=1
root_id=1
head_id=2    头面部
body_id=3    身体结构
wear_id=4    穿戴
```

启动命令：

```bash
EIDORY_DATA_DIR=/tmp/eidory-ui-node-retention-test ./run_eidory.sh
```

可见 UI 测试进度：
- 已成功启动源码版 Eidory。
- 已确认必须用截图原始像素坐标点击，普通点坐标会偏。
- 有效示例：
  - 点击右侧 AI tab：`/opt/homebrew/bin/cliclick c:1190,392`
  - 点击左侧创作项目区展开三角：`/opt/homebrew/bin/cliclick c:178,1028`
- 截图路径包括：
  - `/tmp/eidory-ui-node-retention-step4.png`
  - `/tmp/eidory-ui-node-retention-step6.png`
  - `/tmp/eidory-ui-node-retention-step7.png`
- UI 测试被用户中断，尚未完成输入、切换、切回验证。

已停止残留测试进程 PID `9052`。

## Pending Work

下一轮建议按这个顺序做：

1. 完成真实可见界面测试
   - 使用临时数据目录，不碰真实图库：
     ```bash
     EIDORY_DATA_DIR=/tmp/eidory-ui-node-retention-test ./run_eidory.sh
     ```
   - 在可见界面里：
     1. 打开 AI 面板；
     2. 选择左侧「CU节点保存测试」项目；
     3. 选择右侧「头面部」；
     4. 在「当前节点说明」输入：`脏辫，右脸烧伤，蓝色义眼`；
     5. 切到「身体结构」；
     6. 切回「头面部」；
     7. 确认文本仍在界面里；
     8. 用 SQLite 验证：
        ```bash
        sqlite3 /tmp/eidory-ui-node-retention-test/eidory.sqlite3 \
          "select id,title,note,search_query from creative_nodes order by id;"
        ```

2. 如果真实界面仍清空
   - 不要继续只改 AI prompt。
   - 优先加临时调试日志，记录：
     - `_queue_creative_node_auto_save`
     - `_capture_visible_creative_node_editor`
     - `_save_current_creative_node_editor_snapshot`
     - `_sync_creative_node_panel`
     - `_on_creative_node_selection_changed`
   - 日志必须包含 node_id、当前 editor 文本、store 写入结果、触发事件类型。
   - 用环境变量控制，例如 `EIDORY_DEBUG_CREATIVE_NODE_SAVE=1`，避免污染正常用户日志。

3. 修复 AI 补全质量
   - 当前核心问题可能不是 prompt，而是手写节点内容丢失，导致 AI 没拿到约束。
   - 保存问题确认解决后，再测：
     - 「当前节点信息补全」
     - 「补全所有节点信息」
   - 测试必须跨模板：
     - 故事性插画
     - 角色设计
     - 物件/其他模板如存在
   - 每个测试至少手写不同节点内容，例如：
     - 头面部：`脏辫，右脸烧伤，蓝色义眼`
     - 身体结构：`左臂透明机械骨架，背部外露管线`
     - 穿戴：`破旧白大褂，蓝色袖标`
   - 判定标准：补全后原有关键词必须出现在节点说明或以明确同义视觉特征出现，且不得矛盾。

4. 重新打包替换 App
   - 只有真实 UI 测试通过后再做：
     ```bash
     ./scripts/build_macos_app.sh
     pkill -x Eidory || true
     rm -rf /Applications/Eidory.app
     cp -R dist/Eidory.app /Applications/Eidory.app
     xattr -dr com.apple.quarantine /Applications/Eidory.app || true
     ```

## Risks / Caveats

- 用户明确不接受“后台测试通过”作为完成证明。必须可见界面测试。
- `computer-use` 全量读取 Qt UI 树会触发崩溃，不能继续硬开全树。
- `/Applications/Eidory.app` 目前不是最新源码构建。
- 临时测试库可随时删：`/tmp/eidory-ui-node-retention-test`。
- 不要操作真实图库数据库，除非用户明确要求。
- 工作区有多处未提交修改，不能随便回滚。

## Suggested Skills Next

- `build-macos-apps:build-run-debug`
- `build-macos-apps:packaging-notarization`
- `computer-use:computer-use`，但不要全量开启 Qt Accessibility；优先截图/点击或窄化 AX 暴露。
