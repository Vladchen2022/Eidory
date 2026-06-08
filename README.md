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
7. Use the search bar with `语义`, `关键词`, `相似图`, or `颜色`.

Eidory indexes files in place. Importing means “record these files in Eidory and
assign them to the selected logical folder”; it does not copy the source files.

## Search

Search modes:

- `关键词`: searches filename, tag, note, and stored text fields.
- `语义`: encodes the text query and searches image embeddings.
- `相似图`: searches by the selected image embedding.
- `颜色`: searches by dominant color similarity.

Search logic:

- `重新搜索`: search the whole current scope again.
- `在结果中搜`: narrow inside current results.
- `合并结果`: search again and union the new results with current results.

The minimum-similarity slider controls how many semantic/color/probe results are
kept. Lower values are broader; higher values are stricter.

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

## AI Semantic Probes

Open the right sidebar `AI` tab:

1. Write one sentence describing the creative topic.
2. Add optional context such as time, weather, lighting, mood, era, or art direction.
3. Click `生成语义探针`.
4. Select up to 7 probes.
5. Click `保存并搜索` to search mixed reference results.
6. Save chosen results into inspiration projects for later review.

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
