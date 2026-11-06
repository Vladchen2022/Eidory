"""Package the cached retrieval model, including its remote-code dependencies.

No model download or inference is performed while exporting or restoring.
Logical model IDs in the vector table deliberately remain unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


REPOSITORIES = (
    'jinaai/jina-clip-v2', 'jinaai/jina-clip-implementation',
    'jinaai/jina-embeddings-v3', 'jinaai/xlm-roberta-flash-implementation',
)


def cached_model_files(data_dir: Path):
    portable = data_dir / 'models' / 'huggingface'
    hub = (portable / 'hub' if (portable / 'model-manifest.json').exists() else
           Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface'))/'hub')))
    files, revisions = [], {}
    for repo in REPOSITORIES:
        root = hub / ('models--' + repo.replace('/', '--'))
        ref = root / 'refs' / 'main'
        if not ref.is_file():
            raise ValueError(f'本机未完整缓存检索模型依赖 {repo}。请先完成一次语义搜索，或取消携带检索模型。')
        revision = ref.read_text().strip()
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError(f'模型版本记录无效：{repo}')
        snapshot = root / 'snapshots' / revision
        if not snapshot.is_dir():
            raise ValueError(f'缺少模型快照：{repo}')
        revisions[repo] = revision
        files.append((ref, str(Path('models/huggingface/hub') / ref.relative_to(hub))))
        for file in sorted(snapshot.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:
                # Dereference HF cache symlinks into ordinary, portable files.
                files.append((file, str(Path('models/huggingface/hub') / file.relative_to(hub))))
    primary = hub / 'models--jinaai--jina-clip-v2' / 'snapshots' / revisions[REPOSITORIES[0]]
    for name in ('model.safetensors', 'config.json', 'custom_st.py', 'modules.json', 'tokenizer.json'):
        if not (primary / name).is_file():
            raise ValueError(f'检索模型缺少文件：{name}')
    return files, dict(model_name=REPOSITORIES[0], model_revision='main', dim=512, snapshots=revisions)


def configure_portable_model(data_dir: Path):
    """Call before importing transformers/HF libraries in the new process."""
    variables = ('HF_HOME', 'HF_HUB_CACHE', 'HF_MODULES_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    previous = os.environ.pop('EIDORY_PREVIOUS_HF_ENV', '')
    if previous:
        for key, value in json.loads(previous).items():
            if key in variables:
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    root = data_dir / 'models' / 'huggingface'
    if not (root / 'model-manifest.json').is_file():
        return
    metadata = json.loads((root / 'model-manifest.json').read_text())
    if metadata.get('model_name') != REPOSITORIES[0]:
        raise ValueError('工作区携带的检索模型不受支持')
    os.environ['EIDORY_PREVIOUS_HF_ENV'] = json.dumps({key:os.environ.get(key) for key in variables})
    os.environ.update(HF_HOME=str(root), HF_HUB_CACHE=str(root/'hub'),
                      HF_MODULES_CACHE=str(root/'modules'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1')


def provider_options(data_dir: Path):
    root = data_dir / 'models' / 'huggingface'
    if (root / 'model-manifest.json').is_file():
        return dict(cache_folder=str(root/'hub'), local_files_only=True)
    return {}
