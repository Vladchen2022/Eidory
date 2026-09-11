# Eidory

Eidory is a local-first image and reference library built with Python and PySide6.
It is designed for personal creative reference workflows: importing local image
folders, browsing them quickly, searching by text/image/color, and using an LLM
to generate semantic probes for inspiration.

Current status: early personal-use desktop app. It is usable, but the UI and data
model are still changing.

## What It Does

- Indexes local files in place. Source files are not moved, copied, modified, or deleted.
- Supports image files: JPG, JPEG, PNG, and WebP.
- Supports video files: MP4, MOV, M4V, AVI, MKV, and WebM.
- Generates thumbnail cache under `~/Library/Application Support/Eidory/thumbnails`.
- Stores metadata in SQLite under `~/Library/Application Support/Eidory/eidory.sqlite3`.
- Supports nested library folders, manual tags, favorites, notes, and temporary inspiration projects.
- Searches by keyword, semantic text query, similar image, color, and stacked filters.
- Watches indexed local folders for new, changed, deleted, renamed, or moved files.
- Repairs missing files by relinking a single file or remapping a moved folder prefix.
- Detects exact and near-duplicate image candidates without deleting anything automatically.
- Supports multi-image comparison for final reference judgment.
- Uses local embeddings for image semantic search.
- Uses an LLM provider for AI semantic probes and reference grouping.
- Exports selected files or the whole logical library folder tree.

## What It Does Not Do Yet

- Cloud sync
- AI auto-tagging
- Full chat-based recommendation workflow
- Signed macOS distribution
- Auto-update
- HEIC, GIF, PSD, PDF, or design-source-file management

## Install And Run From Source

Requirements:

- macOS
- Python 3.11 or newer
- Optional: `ffmpeg` for video thumbnails
- Optional: LM Studio, Ollama, OpenAI API, DeepSeek API, or another OpenAI-compatible endpoint for AI features

```bash
git clone https://github.com/Vladchen2022/Eidory.git
cd Eidory
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,app]"
eidory
```

If the `eidory` command is not found, run:

```bash
python -m eidory
```

## Build A Local macOS App

```bash
./scripts/build_macos_app.sh
open dist/Eidory.app
```

The generated app is unsigned. On macOS, the first launch may require
right-clicking `dist/Eidory.app` and choosing `Open`.

## First Use

1. Open Eidory.
2. Create or select a library folder in the left sidebar.
3. Click `导入到当前文件夹` to import images from a disk folder into the selected logical folder.
4. Keep `自动监听文件变化` enabled in the settings panel for local file changes.
5. Click `扫描全部`, `扫描新增`, or `扫描缺失` from the settings panel when you need a manual scan.
6. Click `开始索引` if semantic search has not indexed the imported images yet.
7. Add conditions in the gallery's `保留` / `排除` panel, then click `应用筛选`.

Eidory indexes files in place. Importing means “record these files in Eidory and
assign them to the selected logical folder”; it does not copy the source files.

## Search

The filter panel separates scope, rules, and sorting. Choose the whole library,
one or more folders (including descendants), or freeze the current wall as a
fixed result scope. Selecting a folder in the sidebar changes the scope.

- `保留` groups default to `全部（且）`. For example, airplane plus landscape
  keeps landscape airplane references. With no include conditions, all images
  in the scope are eligible.
- `排除` groups remove matches from those eligible images. Choose `任一（或）`
  to remove red images **or** airplane images; choose `全部（且）` to remove
  only images that match both red and airplane.
- `＋ 组` inside either box adds another group. Exclusion groups remove their
  matches independently; include groups merge their matches. Empty groups have
  no effect.
- Add, edit, remove, or clear conditions in the draft; click `应用筛选` to update
  the wall. `撤销` restores the previous applied query, and `查看排除` previews
  hidden matches without deleting files or library records.

Conditions cover semantic content, keywords, color, similar images, folders,
user tags, file type, orientation, size, status, and existing AI vision tags.
Keywords search filename, path, note, and user tags; use semantic content to
recognize an airplane that is not mentioned in that text. Color describes the
whole image's color tendency, not a particular object's color.

The panel has two permanently visible boxes: `保留` on the left and `排除` on
the right. Every condition occupies one row, with its name and inline strength
slider when applicable. Each box grows and shrinks to fit its own rows. Long
lists scroll within the box to leave space for the gallery; five conditions
remain visible together. Shared scope, sorting, and Apply controls sit outside
the boxes. Scope restrictions and index-coverage details are in the footer.

Each semantic/color/similar-image condition has its own inline `宽松—严格` slider.
After applying the rules, dragging a slider updates the wall and counts while
the mouse is still moving, without another Apply or model search. Loosening an
include condition can restore pictures from the full original scope; loosening
an exclusion condition removes more pictures. One Undo reverses the whole drag.
Changes to rule content, groups, or matching mode remain staged until Apply;
sliders in that pending draft also take effect when those rules are applied.
Strength is relative to that condition's scores across the original scope,
not a recognition probability. Editing another rule does not move its threshold.
Semantic and similar-image conditions only match indexed images; the panel
reports incomplete index coverage. Removing a rule recomputes from the original
scope, so previously filtered images can return.

The preset menu saves and restores scope, include/exclude groups, their all/any
modes, individual strengths, manual exclusions, and sorting. Existing presets
remain readable. Global folder settings that disable search still apply and
are identified in the panel.

## Library Maintenance

Open the right sidebar `设置` tab for maintenance:

- `查看/修复丢失`: lists missing source files, relinks one file, remaps a moved folder, or removes missing records from Eidory only.
- `扫描新增`: finds new or changed files without marking deleted files as missing.
- `扫描缺失`: rescans folders that currently contain missing records.
- `检测重复`: finds exact duplicate files and near-duplicate image candidates. It shows each candidate's folder so the user decides whether the duplicate is intentional.
- `操作历史`: shows batch operations from the current app session.

Automatic file watching marks deleted, renamed, or moved files as missing, but it
does not silently delete records or decide which duplicates should be removed.

Right-click selected images and choose `对比查看` to compare 2-6 images side by
side. Double-click a single image for the existing large preview with zoom,
source-file actions, and next/previous navigation.

## Creative Workspace

The central workspace follows `构思 → 找参考 → 已收集 → 看板`. Gallery images
already collected by any node in the active project show a gold `本项目已选`
badge; hovering lists their current node names. This survives live filtering
and updates when references or projects change. Temporary mouse selection does
not count as collection, and intentional reuse across nodes remains available.
The right
`创作` sidebar keeps the current project and node visible throughout this flow.

- Each project keeps its own brief and extra constraints. Free exploration does
  not overwrite those constraints when switching back to the project.
- AI node expansions appear as editable proposals. Adopt, append a selected
  passage, or discard them; restore the text from before the last adoption.
  Lock confirmed nodes to keep later AI completion from changing them.
- `围绕当前主题完善项目` continues the current idea; `给我一个新创作题目`
  starts a different idea. Batch completion skips locked nodes and proposals
  that have not yet been reviewed.
- `为当前节点找参考` uses the visible search scope and existing filter panel.
  Returning from collected references or the board restores that node's search
  rules, scroll position, and selected candidates.
- Select references and add them to the current node. A reference can have a
  different usage note in each node, without changing its library-wide notes.
- `AI 分组…` previews groups for four or more selected indexed images. Rename,
  move, or merge the suggestions before saving them as reference projects.
- Boards support node sections, text notes, and undo/redo for layout edits.
  `适应全部` includes section headings and notes as well as images.
- `导出创作参考包…` creates a folder with readable HTML, structured JSON,
  original reference files, and board PNGs. The report identifies missing files.

## AI Semantic Probes

Choose `探索灵感方向` in the right `创作` sidebar:

1. Write one sentence describing the creative topic.
2. Add optional context such as time, weather, lighting, mood, era, or art direction.
3. Click `生成语义探针`.
4. Select up to 7 probes.
5. Click `保存并搜索` to search mixed reference results.
6. Adjust the exposed `更多 / 更相近` slider to broaden or narrow the results.
7. Save chosen results, or turn selected directions into nodes of the current
   creative project. Selected images that match each direction are carried into
   the corresponding node.

Default local AI provider:

- LM Studio
- Endpoint: `http://localhost:1234/v1`

You can change the provider in the right sidebar `设置` tab. Supported provider
types are LM Studio, OpenAI API, DeepSeek API, Ollama, and generic
OpenAI-compatible endpoints.

API keys are stored in the local SQLite settings table. They should not be
committed to this repository and are not written into logs by design.

## Semantic Model

The default embedding model is `jinaai/jina-clip-v2`.

Important constraints:

- The first semantic indexing run downloads the model from Hugging Face.
- It uses `trust_remote_code=True`.
- The model license is `cc-by-nc-4.0`, so the current default is intended for
  non-commercial personal use.
- Before commercial use, review the model license and consider replacing the
  default provider.

## Export

Eidory has two export paths:

- `导出选中`: copies the currently selected files to a user-selected folder.
- `导出图库`: copies all non-missing library files into a user-selected folder,
  preserving Eidory's current logical folder tree.

Export never modifies source files. If multiple files have the same name in the
same export target, Eidory automatically creates unique filenames.

Whole-library export is useful before moving machines or rebuilding the app:

1. Click `导出图库`.
2. Choose an empty destination folder.
3. After reinstalling Eidory, import that destination folder by disk directory.

## Data Location

Default data directory:

```text
~/Library/Application Support/Eidory/
```

Main files:

```text
~/Library/Application Support/Eidory/eidory.sqlite3
~/Library/Application Support/Eidory/thumbnails/
```

The settings page includes buttons to open the data directory, back up the
database, restore the database, and run startup checks.

## Run Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

The test suite uses fake embedding providers where possible. It should not need
to download the real semantic model.
PySide6 is pinned to 6.9.3. Newer 6.10/6.11 runtimes intermittently crashed
inside native object bindings during repeated-window regression tests.

## Development Notes

The UI is already broad for this stage. Before adding more large features,
`src/eidory/ui/main_window.py` should be split into smaller controllers or panels.
New AI features should reuse the existing embedding/search services or introduce
isolated services with fake-provider tests.

## License

Eidory's source code is available under the PolyForm Noncommercial License
1.0.0. See [LICENSE](LICENSE).

This means you may read, use, modify, and redistribute the code for
non-commercial purposes, but commercial use is not permitted without a separate
license from the copyright holder. This is a source-available non-commercial
license, not an OSI-approved open source license.

The default embedding model is still governed by its own license. In particular,
`jinaai/jina-clip-v2` is licensed separately and is currently treated by this
project as a non-commercial personal-use default. Eidory's repository license
applies to this repository's code, not to third-party model weights or services.
