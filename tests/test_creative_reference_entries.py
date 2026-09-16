import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt

from test_filter_workflow import app, library, window
from test_creative_workspace import project
from eidory.core.creative_entries import new_entry, encode_entries, decode_entries, search_names
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.llm_provider import LMStudioProvider, LLMProviderError, CreativeNodeNoteSuggestion, parse_creative_node_note_suggestion
from eidory.core.metadata_store import MetadataStore


IDEAS = ({"name": "废土拼装汽车", "visual": "车尾连接回收金属货架"},
         {"name": "手摇水泵", "visual": "用旧管道连接储水桶"})


def accepted(w, node):
    ws = w.creative_workspace
    assert ws.store.stage_entries(node.id, IDEAS)
    ws.refresh()
    ws.accept_note()
    return ws.store.entries(node.id)


def test_structured_parser_keeps_fields_and_rejects_invalid_types():
    result = parse_creative_node_note_suggestion(json.dumps({"additions": [*IDEAS, {"name": [], "visual": "bad"}, IDEAS[0]]}))
    assert result.additions == IDEAS


def test_structure_retry_and_no_prose_fallback():
    provider = LMStudioProvider(model_name="fixture")
    with patch.object(provider, "_chat_completion", side_effect=['{"note":"无结构的说明"}', json.dumps({"additions": IDEAS})]) as chat:
        result, _ = provider.generate_creative_reference_entries(project_brief="废土村庄", node_title="物件")
    assert result.additions == IDEAS and chat.call_count == 2
    with patch.object(provider, "_chat_completion", return_value='{"additions":[{"name":[],"visual":3}]}') as chat:
        with pytest.raises(LLMProviderError, match="原有内容未改动"):
            provider.generate_creative_reference_entries(project_brief="废土村庄", node_title="物件")
    assert chat.call_count == 2


def test_organize_mode_extracts_original_without_object_invention_prompt():
    provider = LMStudioProvider(model_name="fixture")
    with patch.object(provider, "_chat_completion", return_value=json.dumps({"additions": IDEAS})) as chat:
        result, _ = provider.generate_creative_reference_entries(project_brief="废土村庄", node_title="物件",
            current_note="拼装汽车与手摇水泵", completion_mode="organize")
    assert result.additions == IDEAS
    assert "禁止新增、推断或联想" in chat.call_args.kwargs["messages"][1]["content"]


def test_adopt_checkboxes_persist_and_restore_original(window):
    w = window
    _, node = project(w)
    original = w.store.get_creative_node(node.id)
    ws = w.creative_workspace
    assert ws.store.stage_entries(node.id, IDEAS)
    ws.refresh()
    assert w.store.get_creative_node(node.id).note == original.note
    ws.proposal_entries_table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    ws.accept_note()
    rows = ws.store.entries(node.id)
    assert rows["base_note"] == original.note
    assert search_names(rows) == [IDEAS[0]["name"]]
    assert IDEAS[1]["name"] in w.store.get_creative_node(node.id).note
    reopened = CreativeWorkspaceStore(MetadataStore(w.store.database_path))
    assert reopened.entries(node.id) == rows
    ws.restore_note()
    assert ws.store.entries(node.id) is None
    assert w.store.get_creative_node(node.id).note == original.note


def test_rows_search_independently_union_dedup_and_never_search_unchecked(window):
    w = window
    _, node = project(w)
    payload = accepted(w, node)
    images = w.store.list_images(limit=None)
    calls = []
    def search(query, **kw):
        calls.append((query, kw["allowed_image_ids"]))
        matches = images[:2] if query == IDEAS[0]["name"] else images[1:3]
        return SimpleNamespace(images=[replace(im, score=.95) for im in matches], searchable_count=len(images), candidate_limit=len(images))
    w.search_service.semantic_search = search
    w._search_selected_creative_node()
    w._poll_events()
    assert [call[0] for call in calls] == [idea["name"] for idea in IDEAS]
    assert len(w.grid_view.images()) == 3
    assert {im.id for im in w.grid_view.images()} == {im.id for im in images[:3]}
    assert w.filter_workflow.plan().include[0].match_mode == "any"
    slider = w.filter_panel.strength_sliders["include", 0, 1]
    slider.setValue(85)
    w.filter_workflow.finish_strength()
    w.creative_workspace.show_planning()
    w.creative_workspace.entries_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    calls.clear()
    w.creative_workspace.show_candidates()
    w._poll_events()
    assert [call[0] for call in calls] == [IDEAS[1]["name"]]
    assert len(w.grid_view.images()) == 2
    assert w.filter_workflow.plan().include[0].filters[0].score_threshold == 85
    w.creative_workspace.show_planning()
    w.creative_workspace.entries_table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    calls.clear()
    w._search_selected_creative_node()
    assert not calls and "至少勾选" in w.statusBar().currentMessage()


def test_detail_updates_only_target_and_rejects_stale_edits(window):
    w = window
    _, node = project(w)
    ws = w.creative_workspace
    payload = accepted(w, node)
    target = deepcopy(payload["rows"][1])
    target["enabled"] = False
    payload["rows"][1]["enabled"] = False
    assert ws.store.save_entries(node.id, payload)
    assert ws.store.stage_entries(node.id, [{"name": target["name"], "visual": "新增握柄细节"}], mode="details", target=target)
    ws.refresh()
    ws.accept_note()
    updated = ws.store.entries(node.id)
    assert updated["rows"][0] == payload["rows"][0]
    assert updated["rows"][1]["id"] == target["id"]
    assert not updated["rows"][1]["enabled"]
    ws.restore_note()
    assert ws.store.entries(node.id) == payload
    assert ws.store.stage_entries(node.id, IDEAS, mode="details", target=target)
    edited = deepcopy(payload)
    edited["rows"][1]["visual"] = "用户另写的新内容"
    ws.store.save_entries(node.id, edited)
    assert not ws.store.accept_entries(node.id)
    assert ws.store.entries(node.id) == edited


def test_locked_rows_allow_search_selection_but_not_content_changes(window):
    w = window
    _, node = project(w)
    ws = w.creative_workspace
    payload = accepted(w, node)
    ws.toggle_lock(True)
    ws.entries_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not ws.store.entries(node.id)["rows"][0]["enabled"]
    assert not ws.entries_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    changed = deepcopy(payload)
    changed["rows"][0]["name"] = "不可修改"
    assert not ws.store.save_entries(node.id, changed)
    assert not ws.store.stage_entries(node.id, IDEAS)


def test_legacy_database_migrates_without_changing_original_text(library):
    _, store, _ = library
    with store.connect() as conn:
        conn.execute("ALTER TABLE creative_node_workflow DROP COLUMN entries_json")
        conn.execute("ALTER TABLE creative_node_workflow DROP COLUMN proposal_entries_json")
        conn.execute("ALTER TABLE creative_node_versions DROP COLUMN entries_json")
    store.initialize()
    state = CreativeWorkspaceStore(store)
    assert state.node(999)["entries_json"] == ""
    with store.connect() as conn:
        assert {"entries_json", "proposal_entries_json"} <= {r[1] for r in conn.execute("PRAGMA table_info(creative_node_workflow)")}


def test_detail_generation_captures_selected_row_and_does_not_use_checkbox_as_target(window):
    w = window
    _, node = project(w)
    payload = accepted(w, node)
    ws = w.creative_workspace
    ws.entries_table.selectRow(1)
    target = payload["rows"][1]
    captured = []
    def generate(**kwargs):
        captured.append(kwargs)
        return CreativeNodeNoteSuggestion(note="新细节", search_query="", additions=IDEAS[1:], entry_target=kwargs["entry_target"]), "fake"
    with patch.object(w, "_make_llm_provider", return_value=SimpleNamespace(generate_creative_reference_entries=generate)):
        w._generate_creative_children_for_selected_node(completion_mode="details")
        w._poll_events()
    assert captured[0]["entry_target"] == target
    assert ws.store.entries(node.id) == payload
    pending = decode_entries(ws.store.node(node.id)["proposal_entries_json"])
    assert pending["target"]["id"] == target["id"]


def test_node_switch_and_delete_undo_keep_row_ids_and_selection(window):
    w = window
    pid, node = project(w)
    payload = accepted(w, node)
    ws = w.creative_workspace
    ws.entries_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    saved = ws.store.entries(node.id)
    other = next(n for n in w.store.list_creative_nodes(pid) if n.id != node.id)
    w._load_creative_project(pid, select_node_id=other.id)
    assert not ws.entries_table.isVisible()
    w._load_creative_project(pid, select_node_id=node.id)
    assert ws.store.entries(node.id) == saved
    snapshot = w._creative_node_subtree_snapshot(node.id)
    w.store.delete_creative_node(node.id)
    w._creative_node_undo_stack.append({"kind": "deleted", "snapshot": snapshot})
    w._undo_last_creative_node_operation()
    assert ws.store.entries(w.current_creative_node_id) == saved


def test_detail_preserves_existing_description_even_when_ai_omits_it(window):
    w = window
    _, node = project(w)
    ws = w.creative_workspace
    payload = accepted(w, node)
    target = payload["rows"][0]
    assert ws.store.stage_entries(node.id, [{"name": target["name"], "visual": "额外细节"}], target=target)
    assert ws.store.accept_entries(node.id)
    assert ws.store.entries(node.id)["rows"][0]["visual"] == target["visual"] + "\n额外细节"


def test_table_panel_does_not_expand_into_empty_space(window, app):
    w = window
    _, node = project(w)
    accepted(w, node)
    w.resize(2400, 1500)
    w.creative_workspace.show_planning()
    app.processEvents()
    ws = w.creative_workspace
    assert not ws.entries_table.isVisible()
    assert ws.entries_panel.height() < w.creative_flow.idea_cards.sizeHint().height() + 120


def test_discarding_proposal_reenables_generation(window):
    w = window
    pid, _ = project(w)
    node = next(n for n in w.store.list_creative_nodes(pid) if n.title == "物件")
    w._load_creative_project(pid, select_node_id=node.id)
    ws = w.creative_workspace
    assert ws.store.stage_entries(node.id, IDEAS)
    w._sync_creative_node_panel()
    assert not w.generate_creative_children_button.isEnabled()
    ws.discard_note()
    assert w.generate_creative_children_button.isEnabled()
    assert w.generate_creative_details_button.isEnabled()
