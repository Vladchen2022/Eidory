from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eidory.core.inspiration import InspirationTerm, mix_inspiration_search_results
from eidory.core.llm_provider import (
    LMStudioProvider,
    _terms_from_plain_text,
    parse_creative_project_copy_suggestion,
    parse_creative_node_note_suggestion,
    parse_creative_node_suggestions,
    parse_group_name_suggestions,
    parse_inspiration_proposal,
    parse_project_suggestion,
    parse_search_plan_proposal,
)
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


class InspirationTest(unittest.TestCase):
    def test_parse_inspiration_proposal_from_markdown_json(self) -> None:
        proposal = parse_inspiration_proposal(
            """
```json
{
  "questions": ["更偏未来还是复古？"],
  "terms": [
    {"title":"破旧工坊","query":"破旧狭小工坊，机械零件，昏暗灯光","axis":"environment","reason":"环境参考"},
    {"title":"引擎细节","query":"老旧引擎结构，油污金属零件，近景","axis":"object_detail","reason":"机械中心"},
    {"title":"单点台灯","query":"夜晚室内单点台灯，工作台阴影","axis":"lighting","reason":"光源"},
    {"title":"凌乱住处","query":"凌乱出租屋，旧家具，生活痕迹","axis":"environment","reason":"落魄感"},
    {"title":"改装载具","query":"临时改装交通工具，科幻摩托结构","axis":"object_detail","reason":"替代摩托"}
  ]
}
```
            """,
            model_name="fake",
        )

        self.assertEqual(proposal.questions, ["更偏未来还是复古？"])
        self.assertEqual(len(proposal.terms), 5)
        self.assertEqual(proposal.terms[0].title, "破旧工坊")

    def test_parse_inspiration_proposal_with_trailing_text(self) -> None:
        proposal = parse_inspiration_proposal(
            """
下面是结果：
{
  "questions": [],
  "terms": [
    {"title":"潮湿车库","query":"雨夜潮湿车库，机械零件，蓝绿色灯光"},
    {"title":"拆解引擎","query":"拆解中的摩托引擎，油污金属近景"},
    {"title":"旧床旁工位","query":"狭窄住处里的临时工作台，旧床和工具"},
    {"title":"疲惫工程师","query":"疲惫中年工程师，脏外套，低头修理"},
    {"title":"低矮顶灯","query":"低矮顶灯照亮杂乱房间，强烈阴影"}
  ]
}
以上仅供参考。
            """,
            model_name="fake",
        )

        self.assertEqual(len(proposal.terms), 5)
        self.assertEqual(proposal.terms[0].title, "潮湿车库")

    def test_plain_text_terms_fallback(self) -> None:
        terms = _terms_from_plain_text(
            """
1. 潮湿车库：雨夜潮湿车库，机械零件，蓝绿色灯光
2. 拆解引擎：拆解中的摩托引擎，油污金属近景
3. 旧床旁工位：狭窄住处里的临时工作台，旧床和工具
4. 疲惫工程师：疲惫中年工程师，脏外套，低头修理
5. 低矮顶灯：低矮顶灯照亮杂乱房间，强烈阴影
            """
        )

        self.assertEqual(len(terms), 5)
        self.assertEqual(terms[1].query, "拆解中的摩托引擎，油污金属近景")

    def test_plain_text_terms_fallback_from_reasoning_key_value_lines(self) -> None:
        terms = _terms_from_plain_text(
            """
1. title: 环境氛围 | query: 老旧工业风地下室工作间，凌乱工具台，斑驳墙壁，单盏吊灯 | axis: environment | reason: 空间参考
2. title: 光影效果 | query: 丁达尔效应，锥形聚光灯，黑暗边缘光，体积光穿透灰尘 | axis: lighting | reason: 夜晚光线
3. title: 主体细节 | query: 机械师粗糙沾油双手，游标卡尺测量摩托车齿轮 | axis: object_detail | reason: 机械细节
4. title: 材质纹理 | query: 生锈金属表面反射暖光，油污皮革手套，磨损橡胶轮胎 | axis: material | reason: 材质参考
5. title: 构图视角 | query: 低角度仰视摩托车轮廓，前景虚化工具，背景暗角 | axis: composition | reason: 构图参考
            """
        )

        self.assertEqual(len(terms), 5)
        self.assertEqual(terms[0].title, "环境氛围")
        self.assertEqual(terms[0].axis, "environment")

    def test_parse_project_suggestion(self) -> None:
        name, summary = parse_project_suggestion(
            '{"name":"潮湿机械住处","summary":"用于寻找落魄工程师住处和机械细节的参考。"}'
        )

        self.assertEqual(name, "潮湿机械住处")
        self.assertEqual(summary, "用于寻找落魄工程师住处和机械细节的参考。")

    def test_parse_creative_node_suggestions_normalizes_nodes(self) -> None:
        suggestions = parse_creative_node_suggestions(
            """
{
  "nodes": [
    {"title":"凌乱工作台","note":"住处里的维修台、工具和生活痕迹。","search_query":"凌乱工作台 机械零件"},
    {"title":"凌乱工作台","note":"重复节点应丢弃。","search_query":"重复"},
    {"title":"特殊摩托车","note":"复古摩托和改装结构。","search_query":""}
  ]
}
            """
        )

        self.assertEqual([item.title for item in suggestions], ["凌乱工作台", "特殊摩托车"])
        self.assertEqual(suggestions[1].search_query, "特殊摩托车 复古摩托和改装结构。")

    def test_parse_creative_node_note_suggestion(self) -> None:
        suggestion = parse_creative_node_note_suggestion(
            '{"note":"雨夜小巷、潮湿地面和低位霓虹反光。","search_query":"雨夜潮湿小巷 霓虹反光"}'
        )

        self.assertEqual(suggestion.note, "雨夜小巷、潮湿地面和低位霓虹反光。")
        self.assertEqual(suggestion.search_query, "雨夜潮湿小巷 霓虹反光")

    def test_parse_creative_project_copy_suggestion(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "copy_text": "狭窄住处里，疲惫的工程师在冷蓝色台灯下拆解摩托引擎。",
  "nodes": [
    {"title":"地点","note":"狭窄出租屋和临时工作台。","search_query":"狭窄出租屋 临时工作台"},
    {"title":"地点","note":"重复应丢弃。","search_query":"重复"}
  ]
}
            """
        )

        self.assertIn("工程师", suggestion.copy_text)
        self.assertEqual(len(suggestion.nodes), 1)
        self.assertEqual(suggestion.nodes[0].title, "地点")

    def test_parse_creative_project_copy_suggestion_unwraps_chat_response_content(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "choices": [
    {
      "message": {
        "content": "{\\"copy_text\\": \\"雨夜站台上，维修师背对远处灯箱，湿地反光压出孤独感。\\", \\"nodes\\": []}"
      }
    }
  ]
}
            """
        )

        self.assertIn("雨夜站台", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_accepts_plain_content_text(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "content": "冷色维修棚里，主角低头检查满是划痕的车体，背景灯牌被雨雾晕开。"
}
            """
        )

        self.assertIn("维修棚", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_accepts_plain_response_text(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            "雨夜维修棚里，灯箱和积水反光包围着沉默的维修师。"
        )

        self.assertIn("维修棚", suggestion.copy_text)
        self.assertEqual(suggestion.nodes, [])

    def test_parse_creative_project_copy_suggestion_extracts_thinking_draft(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
Here's a thinking process:

1. **Analyze User Input:**
- **Project Theme:** 维修师在屋前的小院里教两个学徒修理一辆汽车

3. **Synthesize & Draft:**
傍晚时分，低矮平房前的水泥小院被夕阳余晖和暖黄工作灯切开。维修师俯身指向敞开的汽车引擎舱，机械义眼映出冷蓝霓虹；两名学徒穿着沾满油污的围裙，半蹲在砖石地面旁记录零件拆装。生锈底盘与发光管线交织，旧轮胎、零件箱、晾衣绳和斑驳外墙共同撑起未来市井的日常质感。

4. **Character Count Check:**
Total: about 220 Chinese characters.
            """
        )

        self.assertTrue(suggestion.copy_text.startswith("傍晚时分"))
        self.assertIn("未来市井", suggestion.copy_text)
        self.assertNotIn("thinking process", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_extracts_partial_json_copy_text(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "copy_text": "暮色四合，低矮平房的院落被冷蓝环境光与暖黄工作灯切割出明暗交界。维修师站在掀开的引擎盖旁，向两个学徒指点底盘节点，金属反光与生活痕迹交织
            """
        )

        self.assertTrue(suggestion.copy_text.startswith("暮色四合"))
        self.assertNotIn("copy_text", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_strips_english_prefix(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            "Start with the low-angle composition and time/lighting: 低角度仰视镜头切入黄昏时分的小院，屋檐灯与暖黄工作探照灯交织出斑驳光影。"
        )

        self.assertTrue(suggestion.copy_text.startswith("低角度仰视"))
        self.assertNotIn("Start with", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_accepts_choice_text(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "choices": [
    {
      "text": "雾气压低的车站外，维修师站在湿亮柏油路边检查引擎盖内的冷蓝光源，远处灯箱和人群剪影把雨夜拉出孤独层次。"
    }
  ]
}
            """
        )

        self.assertIn("维修师", suggestion.copy_text)
        self.assertIn("雨夜", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_uses_reasoning_when_content_is_empty(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "choices": [
    {
      "message": {
        "content": "",
        "reasoning": "分析：先确认节点。\\n最终文案：黄昏小院里，维修师俯身指向拆开的汽车底盘，两名学徒围在暖黄工作灯旁记录步骤，冷蓝霓虹从低矮屋檐下渗入，旧零件、油渍水泥地和晾衣绳共同形成生活化的未来市井感。"
      }
    }
  ]
}
            """
        )

        self.assertTrue(suggestion.copy_text.startswith("黄昏小院"))
        self.assertNotIn("分析", suggestion.copy_text)

    def test_parse_creative_project_copy_suggestion_accepts_description_field(self) -> None:
        suggestion = parse_creative_project_copy_suggestion(
            """
{
  "description": "低矮棚屋前，主角在雨后的泥地上整理拆下的车门和线路，两名助手搬来工具箱，背景中的暖光窗户和冷色路灯形成清楚的明暗对比。"
}
            """
        )

        self.assertIn("低矮棚屋", suggestion.copy_text)

    def test_generate_creative_project_copy_requests_plain_text(self) -> None:
        class FakeProvider(LMStudioProvider):
            def __init__(self) -> None:
                super().__init__(model_name="fake-model")
                self.calls: list[dict[str, object]] = []

            def _chat_completion(self, **kwargs: object) -> str:  # type: ignore[override]
                self.calls.append(kwargs)
                return "雨夜维修棚里，灯箱和积水反光包围着沉默的维修师。"

        provider = FakeProvider()

        suggestion, model_name = provider.generate_creative_project_copy(
            project_brief="雨夜维修棚",
            nodes=[{"title": "地点", "path": "项目 / 地点", "note": "潮湿维修棚", "search_query": ""}],
            language="zh",
        )

        self.assertEqual(model_name, "fake-model")
        self.assertIn("维修棚", suggestion.copy_text)
        self.assertEqual(suggestion.nodes, [])
        self.assertEqual(provider.calls[0]["prefer_json"], False)
        messages = provider.calls[0]["messages"]
        self.assertIsInstance(messages, list)
        self.assertIn("只输出最终文案正文", messages[1]["content"])  # type: ignore[index]

    def test_parse_search_plan_proposal_normalizes_filters(self) -> None:
        proposal = parse_search_plan_proposal(
            """
{
  "questions": ["更偏室内还是室外？"],
  "terms": [
    {"title":"破旧工坊","query":"破旧狭小工坊，机械零件，昏暗灯光"},
    {"title":"引擎细节","query":"老旧引擎结构，油污金属零件，近景"},
    {"title":"单点台灯","query":"夜晚室内单点台灯，工作台阴影"},
    {"title":"凌乱住处","query":"凌乱出租屋，旧家具，生活痕迹"},
    {"title":"改装载具","query":"临时改装交通工具，科幻摩托结构"}
  ],
  "filters": [
    {"field":"scene_location","value":"indoors","optional":false,"reason":"住处或工坊"},
    {"field":"weather","value":"clear","optional":true,"reason":"如果要晴朗外部光"},
    {"field":"shot_scale","value":"full_shot","optional":false,"reason":"人物和环境关系"},
    {"field":"occupation","value":"engineer","optional":false,"reason":"非法字段应丢弃"}
  ]
}
            """,
            model_name="fake",
        )

        self.assertEqual(proposal.questions, ["更偏室内还是室外？"])
        self.assertEqual(len(proposal.terms), 5)
        self.assertEqual(
            [(item.field, item.value, item.optional) for item in proposal.filters],
            [
                ("scene_location", "indoor", False),
                ("weather", "sunny", True),
                ("shot_scale", "long", False),
            ],
        )

    def test_parse_group_name_suggestions_fills_missing_groups(self) -> None:
        suggestions = parse_group_name_suggestions(
            '{"groups":[{"name":"破旧工坊","summary":"昏暗室内和工作台参考。"}]}',
            expected_count=2,
        )

        self.assertEqual(suggestions[0].name, "破旧工坊")
        self.assertEqual(suggestions[1].name, "参考组 2")

    def test_mix_inspiration_results_round_robins_and_records_matches(self) -> None:
        engine = InspirationTerm(title="引擎细节", query="engine")
        room = InspirationTerm(title="破旧住处", query="room")
        first = self._image(1, score=0.9)
        shared_from_engine = self._image(2, score=0.7)
        shared_from_room = self._image(2, score=0.8)
        third = self._image(3, score=0.6)

        result = mix_inspiration_search_results(
            [
                (engine, [first, shared_from_engine]),
                (room, [shared_from_room, third]),
            ]
        )

        self.assertEqual([image.id for image in result.images], [1, 2, 3])
        self.assertEqual(result.images[1].score, 0.8)
        self.assertEqual(
            [match.term_title for match in result.matches_by_image_id[2]],
            ["引擎细节", "破旧住处"],
        )

    def test_inspiration_project_terms_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()

            project_id = store.create_inspiration_project(
                title="机械工程师",
                brief="落魄机械工程师",
                answers="雨夜",
                questions=["更偏未来还是复古？"],
                provider_name="lm_studio",
                model_name="fake",
                terms=[
                    InspirationTerm(title="破旧工坊", query="破旧工坊"),
                    InspirationTerm(title="引擎细节", query="老旧引擎"),
                ],
                selected_titles={"引擎细节"},
            )

            selected = store.inspiration_terms_for_project(project_id, selected_only=True)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].title, "引擎细节")
            self.assertTrue(selected[0].selected)

            projects = store.list_inspiration_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].brief, "落魄机械工程师")
            self.assertEqual(projects[0].answers, "雨夜")
            self.assertEqual(projects[0].questions, ["更偏未来还是复古？"])
            self.assertEqual(projects[0].term_count, 2)
            self.assertEqual(projects[0].selected_count, 1)

            self.assertTrue(
                store.update_inspiration_project_selection(
                    project_id,
                    selected_titles={"破旧工坊"},
                )
            )
            selected = store.inspiration_terms_for_project(project_id, selected_only=True)
            self.assertEqual([term.title for term in selected], ["破旧工坊"])
            self.assertEqual(store.get_inspiration_project(project_id).selected_count, 1)
            self.assertTrue(store.delete_inspiration_project(project_id))
            self.assertEqual(store.list_inspiration_projects(), [])

    @staticmethod
    def _image(image_id: int, *, score: float | None = None) -> ImageItem:
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=f"/tmp/{image_id}.jpg",
            file_name=f"{image_id}.jpg",
            file_ext=".jpg",
            file_size=100,
            width=100,
            height=100,
            created_at=None,
            modified_at=None,
            modified_time_ns=image_id,
            imported_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            thumbnail_path=None,
            thumbnail_status="ready",
            embedding_status="ready",
            is_missing=False,
            is_favorite=False,
            note=None,
            score=score,
        )


if __name__ == "__main__":
    unittest.main()
