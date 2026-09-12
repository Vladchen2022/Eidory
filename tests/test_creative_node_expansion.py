import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_filter_workflow import app, library, window

from eidory.core.llm_provider import (
    LLMProviderError, LMStudioProvider, _creative_node_note_response_format,
    parse_creative_node_note_suggestion,
)
from eidory.ui.main_window import _merge_creative_user_note_with_search_query


CONTEXT = dict(
    project_brief="赛博朋克贫民住所室内",
    project_extra="供电不稳定，居民回收电子废品；没有昂贵新设备",
    project_outline="- 世界观：不允许使用魔法\n- 物件：破旧的床",
    node_title="物件",
    node_path="项目 / 物件",
    current_note="破旧的床",
    imagination_level=90,
)


def test_structured_ideas_are_kept_as_separate_named_lines():
    result = parse_creative_node_note_suggestion(json.dumps({
        "additions": [
            {"name": "元件分拣架", "visual": "抽屉分类存放拆下的电路板和接口。"},
            {"name": "手摇通风扇", "visual": "手动带动叶片，为狭窄工作区通风。"},
            {"name": "元件分拣架", "visual": "重复条目不应再次显示。"},
        ],
        "search_query": "赛博朋克 元件分拣架 手摇通风扇",
    }, ensure_ascii=False))
    assert result.note.splitlines() == [
        "元件分拣架: 抽屉分类存放拆下的电路板和接口。",
        "手摇通风扇: 手动带动叶片，为狭窄工作区通风。",
    ]


def test_empty_structured_ideas_cannot_be_replaced_with_search_text():
    with pytest.raises(LLMProviderError, match="没有提出新的参考想法"):
        parse_creative_node_note_suggestion('{"additions":[],"search_query":"破旧的床与赛博朋克房间"}')


@pytest.mark.parametrize("level,limit", [(0,2), (20,2), (21,3), (50,4), (73,6), (100,8)])
def test_response_contract_limits_additions_for_slider_level(level, limit):
    schema = _creative_node_note_response_format(level)["json_schema"]["schema"]
    assert schema["properties"]["additions"]["maxItems"] == limit
    assert "additions" in schema["required"]


def test_retry_keeps_world_constraints_and_expansion_goal():
    provider = LMStudioProvider(model_name="fake")
    copy_only = json.dumps(dict(note="破旧的床", search_query="破旧的床"), ensure_ascii=False)
    expanded = json.dumps(dict(
        note="破旧的床\n手摇供电箱：停电时给回收终端供电。",
        search_query="赛博朋克 旧床 手摇供电箱 回收终端",
    ), ensure_ascii=False)
    with patch.object(provider, "_chat_completion", side_effect=[copy_only, expanded]) as chat:
        result, _ = provider.generate_creative_node_note(**CONTEXT)
    assert chat.call_count == 2
    retry = chat.call_args.kwargs["messages"][1]["content"]
    assert "不允许使用魔法" in retry
    assert "建议新增 5–8 项" in retry
    assert "物件/道具节点" in retry
    assert "只保留原有核心目标" in retry
    assert result.note.startswith("破旧的床")
    assert "手摇供电箱" in result.note
    assert "\n手摇供电箱" in result.note


def test_copy_only_answers_raise_instead_of_fabricating_stock_prose():
    provider = LMStudioProvider(model_name="fake")
    copy_only = json.dumps(dict(note="破旧的床", search_query="破旧的床"), ensure_ascii=False)
    with patch.object(provider, "_chat_completion", return_value=copy_only) as chat:
        with pytest.raises(LLMProviderError, match="原有说明已保留"):
            provider.generate_creative_node_note(**CONTEXT)
    assert chat.call_count == 2


def test_batch_does_not_disguise_failed_node_as_a_success():
    provider = LMStudioProvider(model_name="fake")
    payload = {"nodes": [
        dict(node_id="1", note="破旧的床", search_query="旧床"),
        dict(node_id="2", note="百叶窗间隙投射分段光带，墙面反射形成床侧补光。", search_query="百叶窗光带 墙面反射补光"),
    ]}
    with patch.object(provider, "_chat_completion", return_value=json.dumps(payload, ensure_ascii=False)):
        result, _ = provider.generate_creative_node_notes(
            project_brief=CONTEXT["project_brief"],
            nodes=[dict(node_id="1", node_title="物件", current_note="破旧的床"),
                   dict(node_id="2", node_title="光线", current_note="")],
        )
    assert set(result) == {"2"}


def test_new_search_targets_survive_a_long_existing_note():
    old = "拼装汽车停在村口，有已经确认的复杂背景和构图关系。" * 20
    query = _merge_creative_user_note_with_search_query(old, "废土 手压水泵 蓄水罐 回收零件架", "物件")
    assert len(query) <= 240
    assert query.startswith("废土 手压水泵 蓄水罐 回收零件架")


def test_query_still_keeps_existing_traits_and_filters_known_conflicts():
    query = _merge_creative_user_note_with_search_query("秃头，老人，老年斑", "年轻 浓密头发 义眼 金属接口", "头面部")
    assert all(term in query for term in ("秃头", "老人", "老年斑", "义眼", "金属接口"))
    assert "年轻" not in query
    assert "浓密头发" not in query


def test_copy_only_ui_result_leaves_note_and_proposal_unchanged(window):
    w = window
    project_id = w.store.create_creative_project(title="补全测试", brief="赛博朋克室内")
    node_id = w.store.create_creative_node(
        project_id=project_id, parent_id=w.store.creative_root_node_id(project_id),
        title="物件", note="破旧的床", search_query="旧床",
    )
    w.creative_workspace.store.save_node(node_id, proposal_note="手摇供电箱", proposal_query="手摇供电箱")
    result = w._apply_creative_node_note_suggestion(
        node_id, "破旧的床", SimpleNamespace(note="破旧的床", search_query="旧床"), stage_only=True,
    )
    assert result is None
    assert w.store.get_creative_node(node_id).note == "破旧的床"
    assert w.creative_workspace.store.node(node_id)["proposal_note"] == "手摇供电箱"


@pytest.mark.parametrize('mode', ['objects', 'details'])
def test_retry_keeps_the_selected_mode(mode):
    p = LMStudioProvider(model_name='fake')
    payload = json.dumps(dict(additions=[dict(name='床架接缝', visual='床架焊接点有磨平后留下的银灰色痕迹。')], search_query='床架 焊缝'), ensure_ascii=False)
    with patch.object(p, '_chat_completion', side_effect=['{"note":"破旧的床"}', payload]) as call:
        p.generate_creative_node_note(**CONTEXT, completion_mode=mode)
    for request in call.call_args_list:
        prompt = request.kwargs['messages'][1]['content']
        assert ('当前模式：补充细节' in prompt) == (mode == 'details')
        assert ('每项名称必须是一件可画出来的东西' in prompt) == (mode == 'objects')


def test_time_node_rejects_objects_before_calling_model():
    p = LMStudioProvider(model_name='fake')
    with patch.object(p, '_chat_completion') as call:
        with pytest.raises(LLMProviderError, match='补充细节'):
            p.generate_creative_node_note(project_brief='夜晚的村庄', node_title='时间', node_path='项目 / 时间', completion_mode='objects')
    call.assert_not_called()


def test_two_buttons_select_modes_and_respect_node_lock_and_busy(window):
    w = window
    pid = w.store.create_creative_project(title='模式测试', brief='赛博朋克室内')
    root = w.store.creative_root_node_id(pid)
    props = w.store.create_creative_node(project_id=pid, parent_id=root, title='物件', note='破旧的床')
    lighting = w.store.create_creative_node(project_id=pid, parent_id=root, title='光线', note='只用窗外街灯')
    w._load_creative_project(pid, select_node_id=props, show_board=False)
    captured = []
    provider = SimpleNamespace(generate_creative_node_note=lambda **kw: (captured.append(kw) or SimpleNamespace(note='床架焊缝有细小的磨平痕迹', search_query='床架焊缝'), 'fake'))
    with patch.object(w, '_make_llm_provider', return_value=provider), patch.object(w, '_start_background_task', side_effect=lambda run, **kw: run()):
        w.generate_creative_children_button.click()
        assert captured[-1]['completion_mode'] == 'objects'
        assert not w.generate_creative_details_button.isEnabled()
        w._set_creative_node_completion_busy(False)
        w.generate_creative_details_button.click()
        assert captured[-1]['completion_mode'] == 'details'
    w._set_creative_node_completion_busy(False)
    w.creative_workspace.store.save_node(props, locked=1)
    w._sync_creative_node_panel()
    assert not w.generate_creative_children_button.isEnabled()
    assert not w.generate_creative_details_button.isEnabled()
    w._load_creative_project(pid, select_node_id=lighting, show_board=False)
    assert not w.generate_creative_children_button.isEnabled()
    assert w.generate_creative_details_button.isEnabled()
    assert '补充细节' in w.generate_creative_children_button.toolTip()


def test_staged_named_suggestion_keeps_first_object_name(window):
    pid = window.store.create_creative_project(title='命名条目测试')
    nid = window.store.create_creative_node(project_id=pid, parent_id=window.store.creative_root_node_id(pid), title='物件', note='破旧的床')
    window._apply_creative_node_note_suggestion(nid, '破旧的床', SimpleNamespace(note='破旧的床\n拼凑工作台: 回收桌面用于分拣电子元件。\n零件收纳箱: 分类存放拆下的小零件。', search_query='拼凑工作台 零件收纳箱'), stage_only=True)
    proposal = window.creative_workspace.store.node(nid)['proposal_note']
    assert '拼凑工作台: 回收桌面' in proposal
    assert proposal.startswith('破旧的床\n')
