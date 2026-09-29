import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from PIL import Image
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from eidory.core.comfy_reference import (
    ComfyClient, GeneratedReferences, build_prompt, local_url, prepare_template,
)
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.creative_entries import encode_entries, new_entry
from eidory.core.comfy_workflows import ReferenceWorkflows, builtin_template


@pytest.fixture
def graph():
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'model.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'encoder.safetensors', 'type': 'qwen_image'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'vae.safetensors'}},
        '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt': 'original', 'negative_prompt': '', 'resolution': 1024}},
        '5': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 1152, 'height': 2048, 'batch_size': 1}},
        '6': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4', 1], 'latent_image': ['5', 0], 'seed': 123, 'steps': 40, 'cfg': 1, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1}},
        '7': {'class_type': 'VAEDecode', 'inputs': {'samples': ['6', 0], 'vae': ['3', 0]}},
        '8': {'class_type': 'SaveImage', 'inputs': {'images': ['7', 0], 'filename_prefix': 'original-output'}},
    }


@pytest.fixture
def generation(library):
    paths, store, _ = library
    project = store.create_creative_project(title='雨夜维修站', brief='赛博朋克城市里的机器人维修站')
    root = store.creative_root_node_id(project)
    node = store.create_creative_node(project_id=project, parent_id=root, title='机器人', note='独眼维修机器人', search_query='机器人')
    return GeneratedReferences(store, paths), project, node


def ready(repo, node, graph):
    job = repo.create(node, '一个独眼维修机器人', prepare_template(graph), 'http://127.0.0.1:8188')
    path = repo.root / 'candidates' / str(job['project_id']) / (job['id'] + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (128, 96), '#557799').save(path)
    thumb = path.with_suffix('.webp')
    Image.open(path).save(thumb)
    job.update(state='ready', file_path=str(path), thumbnail_path=str(thumb), width=128, height=96)
    repo.save(job)
    return job


def test_template_preserves_sampling_and_cannot_write_original_output(graph):
    before = copy.deepcopy(graph)
    template = prepare_template(graph)
    assert graph == before
    assert template['graph']['6']['inputs'] == before['6']['inputs']
    assert template['graph']['8'] == {'class_type': 'PreviewImage', 'inputs': {'images': ['7', 0]}}
    assert template['graph']['4']['inputs']['prompt'] == ''


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost.evil:8188', 'http://user@localhost:8188', 'http://localhost:8188/test'])
def test_reject_nonlocal_endpoints(url):
    with pytest.raises(ValueError):
        local_url(url)


def test_reject_ui_format_and_multiple_outputs(graph):
    with pytest.raises(ValueError, match='API'):
        prepare_template({'nodes': []})
    graph['9'] = copy.deepcopy(graph['8'])
    with pytest.raises(ValueError, match='一个图片输出'):
        prepare_template(graph)


def test_prompt_uses_current_node_and_checked_entries_not_siblings(generation):
    repo, project, node = generation
    store = repo.store
    store.create_creative_node(project_id=project, parent_id=store.creative_root_node_id(project), title='工程师', note='穿白大褂的工程师', search_query='工程师')
    world = store.create_creative_node(project_id=project, parent_id=store.creative_root_node_id(project), title='世界观', note='旧城区低成本科技', search_query='')
    cw = CreativeWorkspaceStore(store)
    cw.save_project(project, extra='恰好一个机器人')
    cw.save_node(node, entries_json=encode_entries({'base_note': '独眼', 'rows': [new_entry('履带底盘'), new_entry('天线', enabled=False)]}))
    text = build_prompt(store, node)
    assert all(part in text for part in ['独眼', '履带底盘', '旧城区低成本科技', '恰好一个机器人'])
    assert '白大褂' not in text and '天线' not in text


def test_timeout_records_identity_and_never_resubmits(generation, graph):
    repo, project, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    with patch.object(ComfyClient, 'validate'), patch.object(ComfyClient, 'request', side_effect=requests.Timeout) as request:
        result = repo.submit(job['id'])
        assert result['state'] == 'unknown'
        assert repo.get(job['id'])['prompt_id'] == job['id']
        repo.submit(job['id'])
        assert request.call_count == 1
    with pytest.raises(ValueError, match='等待'):
        repo.create(node, 'duplicate', prepare_template(graph), 'http://localhost:8188')


def test_reconnect_downloads_only_owned_output_and_is_idempotent(generation, graph):
    repo, project, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    result = {job['id']: {'outputs': {'8': {'images': [{'filename': 'ours.png', 'type': 'temp'}]}, 'other': {'images': [{'filename': 'not-ours.png'}]}}, 'status': {'completed': True}}}
    def download(client, descriptor, target):
        assert descriptor['filename'] == 'ours.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (120, 90), 'green').save(target)
    with patch.object(ComfyClient, 'request', return_value=result), patch.object(ComfyClient, 'download', download):
        recovered = GeneratedReferences(repo.store, repo.paths).reconcile(job['id'])
    assert recovered['state'] == 'ready'
    assert Path(recovered['file_path']).exists()
    with patch.object(ComfyClient, 'request') as request:
        repo.reconcile(job['id'])
        assert not request.called
    assert repo.store.get_image_by_path(recovered['file_path']) is None


def test_cancel_uses_atomic_job_endpoint_only(generation, graph):
    repo, _, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    with patch.object(ComfyClient, 'request', return_value={'cancelled': True}) as request:
        assert repo.cancel(job['id'])['state'] == 'cancelled'
        assert request.call_args.args == ('POST', '/api/jobs/' + job['id'] + '/cancel')


def test_adoption_preserves_target_and_does_not_duplicate(generation, graph, tmp_path):
    repo, project, node = generation
    job = ready(repo, node, graph)
    source = Path(job['file_path'])
    col = repo.store.create_collection('永久生成参考')
    adopted = repo.adopt(job['id'], col, tmp_path / 'permanent')
    assert adopted['state'] == 'adopted'
    assert not source.exists()
    assert Path(adopted['file_path']).exists()
    assert Path(adopted['file_path']).with_suffix('.json').exists()
    assert repo.store.creative_node_image_ids(node) == [adopted['image_id']]
    assert repo.store.image_ids_for_collection(col) == {adopted['image_id']}
    assert repo.adopt(job['id'], col, tmp_path / 'other') == adopted
    assert 'AI 生成参考' in repo.store.get_image(adopted['image_id']).note
    with pytest.raises(ValueError):
        repo.discard(job['id'])


def test_discard_is_recoverable_across_restart(generation, graph):
    repo, project, node = generation
    job = ready(repo, node, graph)
    before = Path(job['file_path']).read_bytes()
    discarded = repo.discard(job['id'])
    assert discarded['state'] == 'discarded'
    assert not Path(job['file_path']).exists()
    restored = GeneratedReferences(repo.store, repo.paths).discard(job['id'], undo=True)
    assert restored['state'] == 'ready'
    assert Path(restored['file_path']).read_bytes() == before
    assert repo.store.get_image_by_path(restored['file_path']) is None


def test_deleted_target_cannot_redirect_to_another_node(generation, graph, tmp_path):
    repo, project, node = generation
    job = ready(repo, node, graph)
    repo.store.delete_creative_node(node)
    col = repo.store.create_collection('候选')
    with pytest.raises(ValueError, match='原节点'):
        repo.adopt(job['id'], col, tmp_path / 'permanent')
    assert Path(job['file_path']).exists()
    assert repo.get(job['id'])['state'] == 'ready'


def test_validation_reports_missing_model(graph):
    template = prepare_template(graph)
    def schema(method, route, **kw):
        if route == '/system_stats':
            return {}
        return {'UNETLoader': {'input': {'required': {'unet_name': [['different.safetensors'], {}]}}}}
    with patch.object(ComfyClient, 'request', side_effect=schema), pytest.raises(ValueError, match='不可用'):
        ComfyClient('http://localhost:8188').validate(template)


def test_single_generation_entry_and_context_guard(window, generation, graph):
    repo, project, node = generation
    w = window
    w._refresh_creative_projects(select_project_id=project)
    w._load_creative_project(project, select_node_id=node)
    w.creative_workspace.show_candidates()
    w.store.set_setting('comfy.template', json.dumps(prepare_template(graph)))
    w.store.set_setting('comfy.profile', 'original')
    c = w.reference_generation
    c.refresh()
    assert c.entry.isVisible() and c.entry.isEnabled()
    assert len([b for b in w.findChildren(QPushButton) if b.text() == '生成参考']) == 1
    c.open()
    assert c.panel.node_id == node
    assert c.panel.width.value() == 1152
    assert '独眼维修机器人' in c.panel.prompt.toPlainText()
    c.panel.close()
    w._load_creative_project(None)
    c.refresh()
    assert not c.entry.isEnabled()


def test_turbo_is_upgrade_default_and_legacy_binding_is_preserved(library, graph):
    _, store, _ = library
    saved = json.dumps(prepare_template(graph))
    store.set_setting('comfy.template', saved)
    profiles = ReferenceWorkflows(store)
    assert profiles.selected() == 'turbo'
    turbo = profiles.template()
    assert turbo['seed_node'] == '13' and turbo['seed_field'] == 'noise_seed'
    assert turbo['graph']['11']['inputs']['strength'] == 1.0
    profiles.select('original')
    assert ReferenceWorkflows(store).selected() == 'original'
    assert profiles.template()['graph']['6']['inputs'] == graph['6']['inputs']
    profiles.import_custom(graph, 'my-workflow.json')
    assert profiles.selected() == 'custom'
    assert [p[0] for p in profiles.options()] == ['turbo', 'original', 'custom']
    assert store.get_setting('comfy.template') == saved


def test_turbo_job_binds_noise_and_preserves_six_step_graph(generation):
    repo, _, node = generation
    template = ReferenceWorkflows(repo.store).template()
    before = copy.deepcopy(template)
    job = repo.create(node, '茶馆里的机器人', template, 'http://localhost:8188', width=1024, height=1024)
    submitted = job['workflow']
    assert submitted['13']['inputs']['noise_seed'] == job['seed']
    assert submitted['13']['inputs']['noise_seed'] != 0
    assert 'seed' not in submitted['6']['inputs']
    assert submitted['4']['inputs']['prompt'] == '茶馆里的机器人'
    assert submitted['15']['inputs']['nodes'] == '1.0, 0.9375, 0.875, 0.75, 0.5, 0.25'
    assert job['workflow_id'] == 'turbo' and '6 步' in job['workflow_name']
    assert template == before
    with patch.object(ComfyClient, 'validate'), patch.object(ComfyClient, 'request', return_value={'prompt_id': job['id']}) as request:
        assert repo.submit(job['id'])['state'] == 'queued'
        assert request.call_args.kwargs['json']['prompt'] == submitted


def test_turbo_rejects_incompatible_size_before_creating_job(generation):
    repo, _, node = generation
    with pytest.raises(ValueError, match='32'):
        repo.create(node, 'robot', builtin_template(), 'http://localhost:8188', width=1000, height=1024)
    assert repo.active() is None


def test_unsupported_advanced_sampler_is_rejected():
    graph = builtin_template()['graph']
    graph['12']['class_type'] = 'CFGGuider'
    with pytest.raises(ValueError, match='BasicGuider'):
        prepare_template(graph)


def test_panel_switch_and_regenerate_use_original_result_snapshot(window, generation, graph):
    repo, project, node = generation
    window._refresh_creative_projects(select_project_id=project)
    window._load_creative_project(project, select_node_id=node)
    window.store.set_setting('comfy.template', json.dumps(prepare_template(graph)))
    c = window.reference_generation
    c.open()
    p = c.panel
    assert p.workflow.currentData() == 'turbo'
    assert (p.width.value(), p.height.value()) == (1024, 1024)
    prompt = p.prompt.toPlainText()
    QTest.mouseClick(p.workflow, Qt.MouseButton.LeftButton)
    assert p.workflow.view().isVisible()
    QTest.keyClick(p.workflow, Qt.Key.Key_Down)
    QTest.keyClick(p.workflow, Qt.Key.Key_Return)
    assert p.workflow.currentData() == 'original'
    assert repo.all(project) == []  # Enter must not submit even a failed job.
    assert (p.width.value(), p.height.value()) == (1152, 2048)
    assert '40 步' in p.sampling.text()
    assert p.prompt.toPlainText() == prompt
    job = ready(repo, node, graph)
    p.refresh_candidates(select_id=job['id'])
    p.workflow.setCurrentIndex(p.workflow.findData('turbo'))
    assert '6 步' in p.sampling.text()
    with patch.object(c, 'generate') as generate:
        c.regenerate()
    snapshot = generate.call_args.kwargs['template']
    assert p.workflow.currentData() == 'history'
    assert '原结果' in p.workflow.currentText()
    assert snapshot['graph']['6']['class_type'] == 'KSampler'
    assert '40 步' in p.sampling.text()
    assert (p.width.value(), p.height.value()) == (1152, 2048)
    assert c.workflows.selected() == 'turbo'
    assert c.template() == snapshot


def test_missing_turbo_does_not_silently_run_original(generation):
    repo, _, node = generation
    job = repo.create(node, 'robot', builtin_template(), 'http://localhost:8188')
    with patch.object(ComfyClient, 'validate', side_effect=ValueError('ViggleTurboLora 不可用')), patch.object(ComfyClient, 'request') as request:
        failed = repo.submit(job['id'])
    assert failed['state'] == 'failed' and 'ViggleTurboLora' in failed['error']
    assert not request.called


def test_resource_hold_does_not_resume_user_paused_index(library):
    import threading
    from eidory.core.embedding_worker import EmbeddingWorker
    from eidory.core.ai_vision_worker import AIVisionWorker
    for cls in (EmbeddingWorker, AIVisionWorker):
        worker = cls.__new__(cls)
        worker._pause_event = threading.Event()
        worker._stop_event = threading.Event()
        worker._generation_ready = threading.Event()
        worker.set_generation_hold(True)
        worker.set_generation_hold(False)
        assert not worker._pause_event.is_set()
        worker.stop()
        assert not worker._wait_ready()


def test_unreachable_server_fails_before_submission(generation, graph):
    repo, _, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    with patch.object(ComfyClient, 'validate', side_effect=requests.ConnectionError), patch.object(ComfyClient, 'request') as request:
        result = repo.submit(job['id'])
    assert result['state'] == 'failed' and '请启动' in result['error']
    assert not request.called


def test_template_placeholder_is_not_a_world_setting(generation):
    from eidory.core.creative_templates import STORY_ILLUSTRATION_TEMPLATE
    repo, project, node = generation
    placeholder = STORY_ILLUSTRATION_TEMPLATE.children[0].note
    repo.store.create_creative_node(project_id=project, parent_id=repo.store.creative_root_node_id(project), title='世界观', note=placeholder, search_query='')
    assert placeholder not in build_prompt(repo.store, node)


def test_adoption_database_failure_keeps_candidate_recoverable(generation, graph, tmp_path):
    repo, project, node = generation
    job = ready(repo, node, graph)
    with pytest.raises(Exception):
        repo.adopt(job['id'], 999999, tmp_path / 'permanent')
    assert Path(job['file_path']).exists()
    assert repo.get(job['id'])['state'] == 'ready'
    assert repo.store.creative_node_image_ids(node) == []
    col = repo.store.create_collection('已修正位置')
    adopted = repo.adopt(job['id'], col, tmp_path / 'permanent')
    assert len(list((tmp_path / 'permanent').glob('*.png'))) == 1
    assert repo.store.creative_node_image_ids(node) == [adopted['image_id']]


def test_download_failure_retries_same_result_without_submission(generation, graph):
    repo, _, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    history = {job['id']: {'outputs': {'8': {'images': [{'filename': 'ours.png', 'type': 'temp'}]}}}}
    with patch.object(ComfyClient, 'request', return_value=history) as request, patch.object(ComfyClient, 'download', side_effect=OSError('disk full')):
        result = repo.reconcile(job['id'])
        assert result['state'] == 'downloading' and 'disk full' in result['error']
        repo.reconcile(job['id'])
        assert all(call.args[0] == 'GET' for call in request.call_args_list)


def test_polling_a_new_result_does_not_steal_selection(window, generation, graph):
    repo, project, node = generation
    first = ready(repo, node, graph)
    window._refresh_creative_projects(select_project_id=project)
    window._load_creative_project(project, select_node_id=node)
    c = window.reference_generation
    c.open()
    assert c.panel.selected_job()['id'] == first['id']
    second = ready(repo, node, graph)
    c.job_updated(second)
    assert c.panel.selected_job()['id'] == first['id']
    assert c.panel.candidates.count() == 2


def test_preview_starts_fitted_and_supports_zoom(window, generation, graph):
    from PySide6.QtWidgets import QDialog
    from eidory.ui.image_preview_dialog import PreviewImageView
    repo, project, node = generation
    ready(repo, node, graph)
    window._refresh_creative_projects(select_project_id=project)
    window._load_creative_project(project, select_node_id=node)
    c = window.reference_generation
    c.open()
    previews = []
    def inspect(dialog):
        view = dialog.findChild(PreviewImageView)
        assert view is not None and view.fit_to_window
        view.set_actual_size()
        assert not view.fit_to_window
        previews.append(view)
        return 0
    with patch.object(QDialog, 'exec', inspect):
        c.preview()
    assert len(previews) == 1


def test_empty_node_scope_clears_other_nodes_result_details(window, generation, graph):
    repo, project, node = generation
    ready(repo, node, graph)
    empty_node = repo.store.create_creative_node(project_id=project, parent_id=repo.store.creative_root_node_id(project), title='地点', note='', search_query='')
    window._refresh_creative_projects(select_project_id=project)
    window._load_creative_project(project, select_node_id=empty_node)
    c = window.reference_generation
    c.open()
    c.panel.show_all.setCurrentIndex(1)
    assert c.panel.selected_job() is not None
    assert '种子' in c.panel.details.text()
    c.panel.show_all.setCurrentIndex(0)
    assert c.panel.selected_job() is None
    assert '还没有生成结果' in c.panel.details.text()
    assert not c.panel.adopt.isEnabled()
