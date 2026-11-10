"""Node-format behavior across all templates, persistence, and AI routing."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from test_filter_workflow import app, library, window
from eidory.core.creative_templates import CREATIVE_TEMPLATES, creative_node_default_format
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.llm_provider import LMStudioProvider, LLMProviderError, CreativeNodeNoteSuggestion, creative_node_supports_objects
from eidory.core.metadata_store import MetadataStore

TEXT_NODES = {
    'story': {'世界观', '时间', '事件', '氛围', '构图'},
    'sceneInterior': {'世界观', '空间功能', '动线与视角', '材质表面', '灯光与色彩', '构图焦点'},
    'sceneExterior': {'世界观', '交通与动线', '天气与光照', '构图焦点'},
    'sceneNatural': {'世界观', '岩石土壤材质', '时间与光照', '路径与危险', '构图焦点'},
    'character': {'世界观', '身份', '身体结构', '头面部'},
    'object': {'世界观', '使用者与用途', '整体轮廓', '材质工艺', '工作状态', '展示尺度'},
}


def create(w, template_id):
    w.creative_template_combo.setCurrentIndex(w.creative_template_combo.findData(template_id))
    pid = w._create_creative_project_from_seed(title='测试-' + template_id, brief='偏远村庄里的修理工作', extra='低技术机械时代，不出现魔法')
    return pid, w.store.list_creative_nodes(pid)


@pytest.mark.parametrize('template', CREATIVE_TEMPLATES, ids=lambda t: t.id)
def test_every_template_node_routes_batch_to_its_own_format(window, template):
    w = window
    pid, nodes = create(w, template.id)
    expected_text = TEXT_NODES[template.id]
    assert {n.title for n in template.root.children if n.content_format == 'text'} == expected_text
    calls = []
    def prose(**kw):
        calls.append(('text', kw['node_title']))
        return CreativeNodeNoteSuggestion(note='这个方向以日常维修需求和相容的视觉关系为依据，保持原先确认的环境与职责。', search_query='低技术机械维修环境', entry_mode='prose'), 'fixture'
    def rows(**kw):
        calls.append(('entries', kw['node_title']))
        return CreativeNodeNoteSuggestion(note='手工维修工具台', search_query='手工工具台', additions=({'name': '手工工具台', 'visual': '用于维修工作的可见支撑结构'},)), 'fixture'
    provider = SimpleNamespace(generate_creative_node_prose=prose, generate_creative_reference_entries=rows)
    w._load_creative_project(pid, select_node_id=nodes[0].id, show_board=False)
    with patch.object(w, '_make_llm_provider', return_value=provider):
        w._generate_all_creative_node_notes()
        w._poll_events()
    assert len(calls) == len(nodes)
    for node in nodes:
        expected = 'text' if node.parent_id is None or node.title in expected_text else 'entries'
        assert (expected, node.title) in calls
        state = w.creative_workspace.store.node(node.id)
        assert bool(state['proposal_entries_json']) == (expected == 'entries')
        assert state['proposal_note']
        assert w.store.get_creative_node(node.id).note == node.note
        w._load_creative_project(pid, select_node_id=node.id, show_board=False)
        w.creative_workspace.show_planning()
        assert w.creative_workspace.format_combo.currentData() == expected
        assert w.creative_workspace.entries_panel.isVisible() == (expected == 'entries')
        if expected == 'text':
            assert w.generate_creative_children_button.text() == '完善说明'
        assert not w.generate_creative_children_button.isEnabled()  # Pending, not overwritable.


def test_single_paragraph_preserves_original_and_keeps_query_separate(window):
    w = window
    pid, nodes = create(w, 'story')
    node = next(n for n in nodes if n.title == '氛围')
    w.store.update_creative_node(node.id, note='平静，不出现恐怖感。', search_query='平静')
    w._load_creative_project(pid, select_node_id=node.id, show_board=False)
    def prose(**kw):
        assert '平静' in kw['current_note']
        assert kw['completion_mode'] == 'auto'
        return CreativeNodeNoteSuggestion(note='漫射天光压低明暗反差，疏朗的色块让空间显得安定。', search_query='柔和漫射光 低对比色块', entry_mode='prose'), 'fixture'
    with patch.object(w, '_make_llm_provider', return_value=SimpleNamespace(generate_creative_node_prose=prose)):
        w.generate_creative_children_button.click()
        w._poll_events()
    ws = w.creative_workspace
    assert not ws.store.entries(node.id)
    assert ws.proposal_input.toPlainText() == '漫射天光压低明暗反差，疏朗的色块让空间显得安定。'
    assert ws.proposal_query.text() == '柔和漫射光 低对比色块'
    ws.accept_note()
    assert w.store.get_creative_node(node.id).note.startswith('平静，不出现恐怖感。')
    w._search_selected_creative_node()
    w._poll_events()
    assert len(w.filter_workflow.plan().include[0].filters) == 1
    ws.show_planning()
    ws.restore_note()
    assert w.store.get_creative_node(node.id).note == '平静，不出现恐怖感。'


def test_format_switch_preserves_rows_and_choices_in_restore_and_restart(window):
    w = window
    pid, nodes = create(w, 'object')
    node = next(n for n in nodes if n.title == '结构拆解')
    w._load_creative_project(pid, select_node_id=node.id, show_board=False)
    ws = w.creative_workspace
    ws.store.stage_entries(node.id, [{'name':'铜制压力阀', 'visual':'手轮控制阀门开合'}])
    ws.refresh()
    ws.accept_note()
    ws.entries_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    before = ws.store.entries(node.id)
    ws.format_combo.setCurrentIndex(ws.format_combo.findData('text'))
    assert ws.store.entries(node.id) is None
    assert '铜制压力阀' in w.store.get_creative_node(node.id).note
    assert ws.store.content_format(node.id, node.title) == 'text'
    reopened = CreativeWorkspaceStore(MetadataStore(w.store.database_path))
    assert reopened.content_format(node.id, node.title) == 'text'
    ws.restore_note()
    assert ws.store.entries(node.id) == before
    assert ws.format_combo.currentData() == 'entries'
    ws.toggle_lock(True)
    assert not ws.format_combo.isEnabled()
    assert not w.generate_creative_children_button.isEnabled()


def test_existing_rows_are_not_automatically_converted_and_custom_children_inherit(window):
    w = window
    pid = w.store.create_creative_project(title='旧项目')
    nid = w.store.create_creative_node(project_id=pid, parent_id=w.store.creative_root_node_id(pid), title='世界观', note='旧说明')
    ws = w.creative_workspace
    ws.store.stage_entries(nid, [{'name': '旧式风车', 'visual': '旧有条目'}])
    ws.store.accept_entries(nid)
    assert ws.store.content_format(nid, '世界观', '项目 / 世界观') == 'entries'
    assert creative_node_default_format('主角', '项目 / 人物 / 主角') == 'entries'
    assert creative_node_default_format('社会制度', '项目 / 世界观 / 社会制度') == 'text'
    assert creative_node_default_format('光照', '项目 / 物件 / 光照') == 'text'
    assert creative_node_default_format('自定义', '项目 / 自定义') == 'text'


def test_only_one_batch_entry_and_skips_locked_and_pending(window):
    w = window
    pid, nodes = create(w, 'story')
    ws = w.creative_workspace
    for node in nodes:
        ws.store.save_node(node.id, locked=1)
    pending = next(n for n in nodes if n.title == '人物')
    ws.store.save_node(pending.id, locked=0, proposal_note='待采纳')
    w._load_creative_project(pid, select_node_id=nodes[0].id)
    buttons = [b.text() for b in w.findChildren(QPushButton)]
    assert buttons.count('批量补全项目节点') == 1
    assert '围绕当前主题完善项目' not in buttons
    with patch.object(w, '_make_llm_provider') as provider:
        w._generate_all_creative_node_notes()
    provider.assert_not_called()
    assert 'AI 正在' not in w.creative_node_status_label.text()


def test_organize_keeps_original_visible_and_saves_separate_ideas(window):
    w = window
    pid, nodes = create(w, 'story')
    node = next(n for n in nodes if n.title == '地点')
    w._load_creative_project(pid, select_node_id=node.id)
    w.creative_flow.show_planning()
    original = '河谷边，干涸的河床\n  ' + '岩壁有断续的水痕。' * 200 + '\n'
    w.creative_node_note_input.setPlainText(original)
    def organize(**kwargs):
        assert kwargs['completion_mode'] == 'organize'
        assert kwargs['current_note'] == original
        return CreativeNodeNoteSuggestion(note='干涸河床', search_query='干涸河床',
            additions=({'name': '干涸河床', 'visual': '裸露的砾石和浅色裂纹'},)), 'fixture'
    ws = w.creative_workspace
    with patch.object(w, '_make_llm_provider', return_value=SimpleNamespace(generate_creative_reference_entries=organize)):
        ws.organize_button.click()
        w._poll_events()
    assert w.store.get_creative_node(node.id).note == original
    ws.accept_note()
    assert ws.store.entries(node.id)['base_note'] == original
    assert w.creative_flow.base_note_view.isVisible()
    assert w.creative_flow.base_note_view.toPlainText() == original
    w.workflow_upgrade.flush()
    assert w.store.get_creative_node(node.id).note.startswith(original + '\n干涸河床: ')


def test_batch_adopts_only_new_prose_after_user_edits(window):
    w = window
    pid, nodes = create(w, 'story')
    text_node = next(n for n in nodes if n.title == '氛围')
    entry_node = next(n for n in nodes if n.title == '地点')
    original_entry = '河谷边，干涸的河床'
    ws = w.creative_workspace
    for node in nodes:
        ws.store.save_node(node.id, locked=int(node.id not in (text_node.id, entry_node.id)))
    w.store.update_creative_node(entry_node.id, note=original_entry)
    assert ws.store.stage_entries(entry_node.id, [{'name': '裸露的石滩', 'visual': '保留原来的边界线'}])
    assert ws.store.accept_entries(entry_node.id)
    old_rows = ws.store.entries(entry_node.id)['rows']
    w._load_creative_project(pid, select_node_id=text_node.id)
    w.creative_node_note_input.setPlainText('清晨，冷色天空。')
    def prose(**kwargs):
        assert '清晨，冷色天空。' in kwargs['current_note']
        return CreativeNodeNoteSuggestion(note='晨光把远处林坡分出几层。', search_query='晨光林坡', entry_mode='prose'), 'fixture'
    def rows(**kwargs):
        assert original_entry in kwargs['current_note']
        return CreativeNodeNoteSuggestion(note='龟裂的浅滩', search_query='龟裂浅滩',
            additions=({'name': '龟裂的浅滩', 'visual': '泥土开裂'},)), 'fixture'
    with patch.object(w, '_make_llm_provider', return_value=SimpleNamespace(generate_creative_node_prose=prose, generate_creative_reference_entries=rows)):
        w._generate_all_creative_node_notes()
        w._poll_events()
    assert w.store.get_creative_node(text_node.id).note == '清晨，冷色天空。'
    assert ws.store.entries(entry_node.id)['rows'] == old_rows
    latest = '清晨，灰蓝色天空。\n  山脊保持平缓。'
    w.creative_node_note_input.setPlainText(latest)
    ws.proposal_input.setPlainText('晨光沿山脊照亮远处林坡。')
    ws.accept_note()
    assert w.store.get_creative_node(text_node.id).note == latest + '\n晨光沿山脊照亮远处林坡。'
    w._load_creative_project(pid, select_node_id=entry_node.id)
    ws.accept_note()
    entries = ws.store.entries(entry_node.id)
    assert entries['base_note'] == original_entry
    assert entries['rows'][:len(old_rows)] == old_rows
    assert entries['rows'][-1]['name'] == '龟裂的浅滩'


def test_prose_schema_retries_rejects_lists_and_never_turns_them_into_rows():
    p = LMStudioProvider(model_name='fixture')
    valid = json.dumps({'note':'地方制度依靠工匠协作维系，低技术条件下的维修能力决定资源分配。', 'search_query':'低技术手工业聚落'})
    with patch.object(p, '_chat_completion', side_effect=['{"additions":[{"name":"新物件","visual":"不应有条目"}]}', valid]) as chat:
        result, _ = p.generate_creative_node_prose(project_brief='工匠村庄', node_title='世界观')
    assert chat.call_count == 2 and not result.additions and result.entry_mode == 'prose'
    schema = chat.call_args.kwargs['response_format']['json_schema']['schema']
    assert set(schema['properties']) == {'note', 'search_query'}
    instructions = '\n'.join(m['content'] for m in chat.call_args.kwargs['messages'])
    assert '当前只负责世界观' in instructions
    assert '不要安排镜头、前中后景、灯光、构图' in instructions
    assert '物件所属时代、技术体系、制造文化' not in instructions
    with patch.object(p, '_chat_completion', return_value='{"note":"1. 风格\\n2. 情绪", "search_query":"光线"}') as chat:
        with pytest.raises(LLMProviderError, match='原有内容未改动'):
            p.generate_creative_node_prose(project_brief='村庄', node_title='氛围')
    assert chat.call_count == 2


def test_format_columns_migrate_without_overwriting_old_notes(library):
    _, store, _ = library
    with store.connect() as conn:
        conn.execute('ALTER TABLE creative_node_workflow DROP COLUMN content_format')
        conn.execute('ALTER TABLE creative_node_versions DROP COLUMN content_format')
    store.initialize()
    with store.connect() as conn:
        assert 'content_format' in {r[1] for r in conn.execute('PRAGMA table_info(creative_node_workflow)')}


def test_template_query_does_not_pollute_prose_and_new_sentences_stay_together(window):
    w = window
    pid, nodes = create(w, 'story')
    node = next(n for n in nodes if n.title == '世界观')
    w.store.update_creative_node(node.id, note='低技术时代。禁止魔法。', search_query=node.note)
    w._load_creative_project(pid, select_node_id=node.id, show_board=False)
    actual = w.store.get_creative_node(node.id)
    assert w._creative_node_ai_constraint_text(actual, w.store.get_creative_project(pid)) == '低技术时代。禁止魔法。'
    result = CreativeNodeNoteSuggestion(note='村落依靠分工协作修理机械。零件沿商路交换，工艺由师徒传授。', search_query='手工业聚落 木质机械', entry_mode='prose')
    w._apply_creative_node_note_suggestion(node.id, actual.note, result, stage_only=True)
    note = w.creative_workspace.store.node(node.id)['proposal_note']
    assert note.replace(' ', '') == result.note
    assert '\n' not in note
    assert '时代、技术水平' not in note


def test_prose_repairs_wrong_query_language():
    p = LMStudioProvider(model_name='fixture')
    bad = json.dumps({'note':'日光在墙上留下轻柔的明暗变化，空间显得平静而疏朗。', 'search_query':'soft light'})
    good = json.dumps({'note':'日光在墙上留下轻柔的明暗变化，空间显得平静而疏朗。', 'search_query':'柔和日光 低对比室内'})
    with patch.object(p, '_chat_completion', side_effect=[bad, good]) as chat:
        result, _ = p.generate_creative_node_prose(project_brief='室内', node_title='氛围')
    assert result.search_query == '柔和日光 低对比室内'
    assert any('搜索语句需使用中文' in m['content'] for m in chat.call_args.kwargs['messages'])
