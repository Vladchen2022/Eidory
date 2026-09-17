"""A named character node must not receive the parent's whole cast."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_filter_workflow import app, library, window
from eidory.core.creative_entries import decode_entries
from eidory.core.creative_tasks import is_character_subject, node_task
from eidory.core.llm_provider import LMStudioProvider, LLMProviderError, CreativeNodeNoteSuggestion


CONTEXT = dict(project_brief='领主骑巨兽，长老巡视，一群奴隶在河滩寻找宝石',
               project_extra='中世纪魔幻世界', node_title='领主', node_path='河滩 / 人物 / 领主',
               current_note='肥胖中年男性，配饰华丽，部落酋长风格',
               sibling_titles=('长老', '奴隶'), imagination_level=100)


def answer(subject='领主', names=('部落领主',)):
    return json.dumps(dict(subject_node=subject, additions=[dict(name=n, visual='厚重金饰形成身份差异，单手控制缰绳。') for n in names],
                           search_query=' '.join(names)), ensure_ascii=False)


@pytest.mark.parametrize('name,path,scoped', [
    ('人物','河滩 / 人物',False), ('领主','河滩 / 人物 / 领主',True),
    ('长老','河滩 / 人物 / 长老',True), ('奴隶','河滩 / 人物 / 奴隶',True),
    ('主角','河滩 / 人物 / 主角',True), ('甲','项目 / 人物 / 敌方 / 甲',True),
    ('穿戴','角色设计 / 穿戴',False), ('头面部','河滩 / 人物 / 领主 / 头面部',False),
    ('书桌','室内 / 陈设与道具 / 书桌',False),
])
def test_subject_scope_inherits_expertise_without_cast_scope(name,path,scoped):
    assert is_character_subject(name,path) == scoped
    if scoped:
        assert node_task(name,path)[0] == '完善当前角色'


@pytest.mark.parametrize('bad', [
    answer(names=('部落领主','白袍长老','河滩奴隶')),
    answer(subject='长老',names=('白袍长老',)),
    answer(names=('白袍长老',)),
    answer(names=('河滩奴隶',)),
    answer(names=('中世纪粗麻兜帽长袍',)),  # a costume is not the current character
    answer(names=('部落领主','部落领主')),  # dedup must not conceal scope violations
    '{"additions":[{"name":"部落领主","visual":"金饰"}]}',
])
def test_scope_failure_retries_with_specific_reason_and_never_accepts_sibling(bad):
    provider=LMStudioProvider(model_name='fixture')
    with patch.object(provider,'_chat_completion',side_effect=[bad,answer()]) as chat:
        result,_=provider.generate_creative_reference_entries(**CONTEXT)
    assert chat.call_count == 2
    assert [r['name'] for r in result.additions] == ['部落领主']
    schema=chat.call_args.kwargs['response_format']['json_schema']['schema']
    assert schema['properties']['additions']['maxItems'] == 1
    assert schema['properties']['subject_node']['enum'] == ['领主']
    prompt=chat.call_args.kwargs['messages'][1]['content']
    assert '同级节点（不属于本次输出目标）：长老、奴隶' in prompt
    assert '已确定三位工程师时，最多三行' not in prompt
    assert '上次未通过校验' in chat.call_args.kwargs['messages'][-1]['content']


def test_two_scope_failures_raise_without_partial_result():
    p=LMStudioProvider(model_name='fixture')
    with patch.object(p,'_chat_completion',return_value=answer(names=('领主','长老','奴隶'))) as chat:
        with pytest.raises(LLMProviderError,match='原有内容未改动'):
            p.generate_creative_reference_entries(**CONTEXT)
    assert chat.call_count == 2


@pytest.mark.parametrize('mode',['objects','details','organize'])
def test_every_entry_mode_keeps_current_subject(mode):
    p=LMStudioProvider(model_name='fixture')
    with patch.object(p,'_chat_completion',return_value=answer()) as chat:
        p.generate_creative_reference_entries(**CONTEXT,completion_mode=mode)
    assert chat.call_args.kwargs['response_format']['json_schema']['schema']['properties']['additions']['maxItems'] == 1


def test_parent_cast_can_still_have_multiple_characters():
    p=LMStudioProvider(model_name='fixture')
    with patch.object(p,'_chat_completion',return_value=answer(names=('领主','长老','奴隶'))) as chat:
        result,_=p.generate_creative_reference_entries(project_brief='三个角色在河滩',node_title='人物',node_path='河滩 / 人物')
    assert len(result.additions) == 3
    assert 'subject_node' not in chat.call_args.kwargs['response_format']['json_schema']['schema']['properties']


def test_prose_child_uses_same_scope_and_retries_wrong_owner():
    p=LMStudioProvider(model_name='fixture')
    bad=json.dumps(dict(subject_node='长老',note='白袍沿着步伐垂落，手持木杖观察河道。',search_query='白袍长老'))
    good=json.dumps(dict(subject_node='领主',note='厚重项链随着骑乘姿态垂在胸前，镶嵌石块强调权力象征。',search_query='肥胖部落领主 金饰'))
    with patch.object(p,'_chat_completion',side_effect=[bad,good]) as chat:
        result,_=p.generate_creative_node_prose(**CONTEXT)
    assert chat.call_count == 2 and result.entry_mode == 'prose'


def test_legacy_single_and_batch_also_keep_subjects_separate():
    p=LMStudioProvider(model_name='fixture')
    with patch.object(p,'_chat_completion',return_value=answer()):
        result,_=p.generate_creative_node_note(**CONTEXT)
        assert len(result.additions) == 1
    with patch.object(p,'_chat_completion',side_effect=[answer(),answer('长老',('白袍长老',))]):
        result,_=p.generate_creative_node_notes(project_brief=CONTEXT['project_brief'],nodes=[
            dict(node_id='1',node_title='领主',node_path='河滩 / 人物 / 领主',sibling_titles=('长老','奴隶')),
            dict(node_id='2',node_title='长老',node_path='河滩 / 人物 / 长老',sibling_titles=('领主','奴隶'))])
    assert result['1'].additions[0]['name']=='部落领主'
    assert result['2'].additions[0]['name']=='白袍长老'


def test_ui_batch_and_single_pass_siblings_and_write_only_requested_node(window):
    w=window
    pid=w.store.create_creative_project(title='节点范围验收',brief=CONTEXT['project_brief'])
    parent=w.store.create_creative_node(project_id=pid,parent_id=w.store.creative_root_node_id(pid),title='人物')
    ids={title:w.store.create_creative_node(project_id=pid,parent_id=parent,title=title,note=title+'原设定')
         for title in ('领主','长老','奴隶')}
    for n in w.store.list_creative_nodes(pid):
        if n.id not in ids.values(): w.creative_workspace.store.save_node(n.id,locked=1)
    w._load_creative_project(pid,select_node_id=ids['领主'],show_board=False)
    calls=[]
    def generate(**kw):
        calls.append(kw)
        return CreativeNodeNoteSuggestion(note='新方案',search_query=kw['node_title'],
            additions=({'name':kw['node_title']+'造型','visual':'该角色的新细节'},)),'fixture'
    with patch.object(w,'_make_llm_provider',return_value=SimpleNamespace(generate_creative_reference_entries=generate)):
        assert w.generate_creative_children_button.text() == '完善“领主”方案'
        w._generate_creative_children_for_selected_node(); w._poll_events()
        assert set(calls[0]['sibling_titles'])=={'长老','奴隶'}
        assert w.creative_workspace.store.node(ids['领主'])['proposal_entries_json']
        assert not w.creative_workspace.store.node(ids['长老'])['proposal_entries_json']
        w._generate_all_creative_node_notes_for_project(pid); w._poll_events()
    assert [c['node_title'] for c in calls] == ['领主','长老','奴隶']
    for title,nid in ids.items():
        rows=decode_entries(w.creative_workspace.store.node(nid)['proposal_entries_json'])['rows']
        assert len(rows)==1 and rows[0]['name']==title+'造型'
        assert w.store.get_creative_node(nid).note==title+'原设定'
