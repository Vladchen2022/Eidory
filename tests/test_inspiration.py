from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eidory.core.inspiration import InspirationTerm, mix_inspiration_search_results
from eidory.core.creative_templates import STORY_ILLUSTRATION_TEMPLATE
from eidory.core.llm_provider import (
    LLMProviderError,
    LMStudioProvider,
    _build_creative_node_note_prompt,
    _build_creative_project_seed_prompt,
    _terms_from_plain_text,
    parse_creative_project_copy_suggestion,
    parse_creative_project_seed_suggestion,
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

    def test_generate_inspiration_terms_accepts_json_from_reasoning_content(self) -> None:
        class FakeProvider(LMStudioProvider):
            def __init__(self) -> None:
                super().__init__(model_name="fake-model")
                self.calls: list[dict[str, object]] = []

            def _chat_completion(self, **kwargs: object) -> str:  # type: ignore[override]
                self.calls.append(kwargs)
                return """
分析：先把主题拆成空间、人物、光线和材质。
{
  "questions": ["自然光从哪一侧进入？"],
  "terms": [
    {"title":"热带集市","query":"tropical market crowded noon realistic modern"},
    {"title":"拥挤人群","query":"dense crowd shopping in market midday"},
    {"title":"摊位遮棚","query":"market stall canvas awning tropical sunlight"},
    {"title":"自然顶光","query":"harsh noon sunlight market shadows"},
    {"title":"现实主义色彩","query":"realistic documentary market scene natural colors"}
  ]
}
                """

        provider = FakeProvider()
        proposal = provider.generate_inspiration_terms(
            brief="来自欧洲的顾客在热带拥挤的集市里采购",
            answers="热带，集市，正午，人群密集，现实主义的现代",
            language="zh",
        )

        self.assertEqual(proposal.model_name, "fake-model")
        self.assertEqual(proposal.questions, ["自然光从哪一侧进入？"])
        self.assertEqual(len(proposal.terms), 5)
        self.assertEqual(proposal.terms[0].title, "热带集市")
        self.assertTrue(provider.calls[0]["allow_reasoning_content"])
        self.assertEqual(provider.calls[0]["reasoning_effort"], "none")

    def test_generate_search_plan_accepts_json_from_reasoning_content(self) -> None:
        class FakeProvider(LMStudioProvider):
            def __init__(self) -> None:
                super().__init__(model_name="fake-model")
                self.calls: list[dict[str, object]] = []

            def _chat_completion(self, **kwargs: object) -> str:  # type: ignore[override]
                self.calls.append(kwargs)
                return """
分析：先生成探针，再补充 AI 场景条件。
{
  "questions": [],
  "terms": [
    {"title":"热带集市","query":"tropical market crowded noon realistic modern"},
    {"title":"拥挤人群","query":"dense crowd shopping in market midday"},
    {"title":"摊位遮棚","query":"market stall canvas awning tropical sunlight"},
    {"title":"自然顶光","query":"harsh noon sunlight market shadows"},
    {"title":"现实主义色彩","query":"realistic documentary market scene natural colors"}
  ],
  "filters": [
    {"field":"scene_location","value":"outdoor","optional":false,"reason":"集市通常在室外或半室外"},
    {"field":"time_of_day","value":"day","optional":false,"reason":"用户指定正午"}
  ]
}
                """

        provider = FakeProvider()
        proposal = provider.generate_search_plan(
            brief="来自欧洲的顾客在热带拥挤的集市里采购",
            answers="热带，集市，正午，人群密集，现实主义的现代",
            language="zh",
        )

        self.assertEqual(proposal.model_name, "fake-model")
        self.assertEqual(len(proposal.terms), 5)
        self.assertEqual([item.field for item in proposal.filters], ["scene_location", "time_of_day"])
        self.assertTrue(provider.calls[0]["allow_reasoning_content"])
        self.assertEqual(provider.calls[0]["reasoning_effort"], "none")

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

    def test_parse_creative_node_note_suggestion_accepts_plain_text(self) -> None:
        suggestion = parse_creative_node_note_suggestion(
            """
节点说明：空间站餐区是一个狭长的模块舱，几名航天员围着固定餐桌吃饭，餐具和小包装食物漂浮在半空，窗外能看到飞船和蓝色地球光。
搜索语句：空间站餐区 航天员吃饭 漂浮餐具 蓝色地球光
            """
        )

        self.assertIn("空间站餐区", suggestion.note)
        self.assertNotIn("节点说明：", suggestion.note)
        self.assertTrue(suggestion.search_query.startswith("空间站餐区"))

    def test_parse_creative_node_note_suggestion_rejects_placeholder_text(self) -> None:
        with self.assertRaises(LLMProviderError):
            parse_creative_node_note_suggestion("...")
        with self.assertRaises(LLMProviderError):
            parse_creative_node_note_suggestion('{"note":"...","search_query":"..."}')

    def test_parse_creative_node_note_suggestion_rejects_reasoning_process(self) -> None:
        with self.assertRaises(LLMProviderError):
            parse_creative_node_note_suggestion(
                "Here's a thinking process: 1. **Analyze User Input:** "
                "Project Theme: 空间站里的几个航天员在吃饭. "
                "2. **Deconstruct Constraints:** Only output JSON."
            )

    def test_parse_creative_project_seed_suggestion(self) -> None:
        suggestion = parse_creative_project_seed_suggestion(
            '{"title":"雨夜补给站","brief":"飞行器驾驶员在雨夜自动售货机旁买饮料",'
            '"extra":"近未来，霓虹，潮湿街道，蓝紫色反光"}'
        )

        self.assertEqual(suggestion.title, "雨夜补给站")
        self.assertIn("飞行器驾驶员", suggestion.brief)
        self.assertIn("蓝紫色反光", suggestion.extra)

    def test_generate_creative_project_seed_uses_relaxed_json_request(self) -> None:
        provider = LMStudioProvider(model_name="fake")
        model_output = (
            '{"title":"雨夜补给站","brief":"飞行器驾驶员在雨夜自动售货机旁买饮料",'
            '"extra":"近未来，霓虹，潮湿街道，蓝紫色反光"}'
        )
        with patch.object(provider, "_chat_completion", return_value=model_output) as chat:
            suggestion, model_name = provider.generate_creative_project_seed(
                template_label="故事性插画",
                template_outline="- 未命名故事性插画\n  - 世界观\n  - 时间",
                language="zh",
            )

        self.assertEqual(model_name, "fake")
        self.assertIn("飞行器驾驶员", suggestion.brief)
        kwargs = chat.call_args.kwargs
        self.assertFalse(kwargs["prefer_json"])
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["temperature"], 1.05)
        self.assertEqual(kwargs["max_tokens"], 900)
        prompt = chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("本次随机创意抽签", prompt)
        self.assertIn("不要默认回到同一种熟悉组合", prompt)
        self.assertIn("随机编号", prompt)
        self.assertIn("/no_think", kwargs["messages"][1]["content"])

    def test_creative_project_seed_prompt_varies_between_calls(self) -> None:
        prompts = {
            _build_creative_project_seed_prompt(
                template_label="故事性插画",
                template_outline="- 未命名故事性插画\n  - 世界观\n  - 时间",
                language="zh",
            )
            for _index in range(4)
        }

        self.assertGreater(len(prompts), 1)
        self.assertTrue(all("本次随机创意抽签" in prompt for prompt in prompts))
        self.assertTrue(all("随机编号" in prompt for prompt in prompts))

    def test_generate_creative_node_note_uses_single_relaxed_json_request(self) -> None:
        provider = LMStudioProvider(model_name="fake")
        model_output = (
            '{"note":"白天的空间站餐区应以舱内柔和人工照明表现日间作息，窗外仍是黑色太空，可见飞船轮廓和冷色星光。",'
            '"search_query":"白天 空间站餐区 黑色太空 舱内人工照明"}'
        )
        with patch.object(provider, "_chat_completion", return_value=model_output) as chat:
            suggestion, model_name = provider.generate_creative_node_note(
                project_brief="空间站里的几个航天员在吃饭",
                project_extra="",
                project_outline="- 空间站里的几个航天员在吃饭\n  - 当前节点：时间",
                node_title="时间",
                current_note="白天，但应该符合太空气氛，空间站外部应该是黑的",
                node_path="空间站里的几个航天员在吃饭 / 时间",
                language="zh",
            )

        self.assertEqual(model_name, "fake")
        self.assertEqual(chat.call_count, 1)
        kwargs = chat.call_args.kwargs
        self.assertFalse(kwargs["prefer_json"])
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["temperature"], 0.35)
        self.assertEqual(kwargs["max_tokens"], 900)
        self.assertIn("黑色太空", suggestion.note)
        self.assertNotIn("thinking process", suggestion.note.lower())

    def test_generate_creative_node_note_retries_without_reasoning_effort_when_unsupported(self) -> None:
        provider = LMStudioProvider(model_name="fake")

        class FakeResponse:
            def __init__(self, status_code: int, content: str = "") -> None:
                self.status_code = status_code
                self._content = content

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": self._content,
                            }
                        }
                    ]
                }

        calls: list[dict[str, object]] = []

        def fake_post(_url: str, **kwargs: object) -> FakeResponse:
            payload = dict(kwargs["json"])  # type: ignore[index]
            calls.append(payload)
            if len(calls) == 1:
                return FakeResponse(400)
            return FakeResponse(
                200,
                '{"note":"舱内保持白天作息的明亮人造光，窗外是黑色太空和飞船剪影。",'
                '"search_query":"空间站 白天 人造光 黑色太空 飞船剪影"}',
            )

        with patch("eidory.core.llm_provider.requests.post", side_effect=fake_post):
            suggestion, _model_name = provider.generate_creative_node_note(
                project_brief="空间站里的几个航天员在吃饭",
                project_extra="",
                project_outline="- 空间站里的几个航天员在吃饭\n  - 当前节点：时间",
                node_title="时间",
                current_note="白天，但应该符合太空气氛，空间站外部应该是黑的",
                node_path="空间站里的几个航天员在吃饭 / 时间",
                language="zh",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].get("reasoning_effort"), "none")
        self.assertNotIn("reasoning_effort", calls[1])
        self.assertIn("黑色太空", suggestion.note)

    def test_generate_creative_node_note_refines_scoped_search_query(self) -> None:
        provider = LMStudioProvider(model_name="fake")
        outputs = {
            "时间": '{"note":"傍晚低角度光线。","search_query":"傍晚 夕阳 暮色 低角度暖光 田野泥地反光 士兵战斗"}',
            "地点": '{"note":"田野泥地环境。","search_query":"中世纪农田 泥泞土地 车辙印 积水反光 傍晚光线 士兵战斗"}',
            "事件": '{"note":"多人近身交锋。","search_query":"中世纪士兵混战 泥泞地面 兵器交锋 近身格斗 动态姿势"}',
            "氛围": '{"note":"烟尘和逆光。","search_query":"中世纪战争 傍晚 逆光 泥泞地面 尘土飞扬 战场氛围感"}',
            "构图": '{"note":"低机位群像。","search_query":"中世纪战场 士兵战斗 单人角色 肖像 低机位广角群像 对角线冲突 前中后景纵深"}',
        }

        def generate(title: str) -> str:
            with patch.object(provider, "_chat_completion", return_value=outputs[title]):
                suggestion, _model_name = provider.generate_creative_node_note(
                    project_brief="几个对立阵营的士兵在战斗",
                    project_extra="中世纪，田野，泥泞，傍晚",
                    project_outline="- 几个对立阵营的士兵在战斗\n  - 世界观\n  - 时间\n  - 地点\n  - 物件\n  - 人物\n  - 事件\n  - 氛围\n  - 构图",
                    node_title=title,
                    current_note="",
                    node_path=f"几个对立阵营的士兵在战斗 / {title}",
                    language="zh",
                )
            return suggestion.search_query

        time_query = generate("时间")
        place_query = generate("地点")
        event_query = generate("事件")
        mood_query = generate("氛围")
        composition_query = generate("构图")

        self.assertIn("傍晚", time_query)
        self.assertNotIn("田野", time_query)
        self.assertNotIn("士兵", time_query)
        self.assertIn("农田", place_query)
        self.assertIn("泥泞土地", place_query)
        self.assertNotIn("中世纪", place_query)
        self.assertNotIn("战斗", place_query)
        self.assertIn("混战", event_query)
        self.assertIn("兵器交锋", event_query)
        self.assertNotIn("中世纪", event_query)
        self.assertNotIn("士兵", event_query)
        self.assertIn("逆光", mood_query)
        self.assertIn("尘土飞扬", mood_query)
        self.assertNotIn("中世纪", mood_query)
        self.assertNotIn("战场", mood_query)
        self.assertIn("低机位广角群像", composition_query)
        self.assertIn("对角线冲突", composition_query)
        self.assertNotIn("中世纪", composition_query)
        self.assertNotIn("士兵", composition_query)
        self.assertNotIn("战斗", composition_query)
        self.assertNotIn("单人", composition_query)
        self.assertNotIn("肖像", composition_query)

    def test_creative_node_note_prompt_prioritizes_user_context(self) -> None:
        prompt = _build_creative_node_note_prompt(
            project_brief="空间站里的几个航天员在吃饭",
            project_extra="未来，科幻，太空，空间站外有飞船和太空设备",
            project_outline="- 空间站里的几个航天员在吃饭\n  - 世界观\n  - 当前节点：时间",
            node_title="时间",
            current_note="必须是晚餐时间，舷窗外有蓝色地球光。",
            node_path="空间站里的几个航天员在吃饭 / 时间",
            language="zh",
        )

        self.assertIn("补充信息", prompt)
        self.assertIn("空间站外有飞船和太空设备", prompt)
        self.assertIn("现有节点说明（如果有，优先级最高）", prompt)
        self.assertIn("不得推翻、删除、替换或违背", prompt)
        self.assertIn("约束和出发点", prompt)
        self.assertIn("合理联想并补充相关的环境、物件、光线", prompt)
        self.assertIn("当前项目节点树", prompt)
        self.assertIn("当前节点：时间", prompt)
        self.assertIn("搜索范围建议", prompt)
        self.assertIn("search_query 的用途是给当前节点找参考图", prompt)
        self.assertIn("search_query 必须使用简体中文短语", prompt)
        self.assertNotIn("围绕这句话扩写", prompt)
        self.assertIn("不要输出“...”“待补充”“同上”", prompt)
        self.assertIn("/no_think", prompt)

    def test_creative_node_note_prompt_broadens_people_search_query_scope(self) -> None:
        prompt = _build_creative_node_note_prompt(
            project_brief="空间站里的几个航天员在吃饭",
            project_extra="未来，科幻，太空，空间站外有飞船和太空设备",
            project_outline="- 空间站里的几个航天员在吃饭\n  - 世界观\n  - 时间\n  - 当前节点：人物\n    - 主角\n    - 次要角色\n  - 事件\n  - 氛围\n  - 构图",
            node_title="人物",
            current_note="",
            node_path="空间站里的几个航天员在吃饭 / 人物",
            language="zh",
        )

        self.assertIn("人物/角色节点", prompt)
        self.assertIn("把参考需求拆开", prompt)
        self.assertIn("航天员、舱内服、轻便宇航服", prompt)
        self.assertIn("更通用的人类参考", prompt)
        self.assertIn("多人围坐吃饭", prompt)
        self.assertIn("造型锚点 + 通用动作/体态", prompt)
        self.assertIn("不要写：空间站、太空舱背景、太空、失重", prompt)

    def test_creative_node_note_prompt_broadens_event_and_composition_search_scope(self) -> None:
        event_prompt = _build_creative_node_note_prompt(
            project_brief="空间站里的几个航天员在吃饭",
            project_extra="未来，科幻，太空",
            project_outline="- 空间站里的几个航天员在吃饭\n  - 当前节点：事件",
            node_title="事件",
            current_note="",
            node_path="空间站里的几个航天员在吃饭 / 事件",
            language="zh",
        )
        composition_prompt = _build_creative_node_note_prompt(
            project_brief="空间站里的几个航天员在吃饭",
            project_extra="未来，科幻，太空",
            project_outline="- 空间站里的几个航天员在吃饭\n  - 当前节点：构图",
            node_title="构图",
            current_note="",
            node_path="空间站里的几个航天员在吃饭 / 构图",
            language="zh",
        )

        self.assertIn("事件/动作节点", event_prompt)
        self.assertIn("几个人一起吃饭", event_prompt)
        self.assertIn("禁止包含时代、地点、时间、阵营身份", event_prompt)
        self.assertIn("禁止写“中世纪、田野、泥泞、傍晚、士兵、战场”", event_prompt)
        self.assertIn("构图节点", composition_prompt)
        self.assertIn("禁止包含时代、地点、人物身份和具体事件", composition_prompt)
        self.assertIn("低机位、俯视、广角群像", composition_prompt)
        self.assertIn("场景类、空间类图片", composition_prompt)
        self.assertIn("禁止把 search_query 写成单个角色", composition_prompt)

    def test_creative_node_note_prompt_keeps_location_anchors_but_broadens_subreferences(self) -> None:
        prompt = _build_creative_node_note_prompt(
            project_brief="空间站里的几个航天员在吃饭",
            project_extra="未来，科幻，太空",
            project_outline="- 空间站里的几个航天员在吃饭\n  - 当前节点：地点",
            node_title="地点",
            current_note="",
            node_path="空间站里的几个航天员在吃饭 / 地点",
            language="zh",
        )

        self.assertIn("地点/环境节点", prompt)
        self.assertIn("自然环境", prompt)
        self.assertIn("人文/建筑环境", prompt)
        self.assertIn("独立道具、武器、载具、家具、工具、器皿和环境附属物", prompt)

    def test_story_illustration_template_has_object_node(self) -> None:
        titles = [node.title for node in STORY_ILLUSTRATION_TEMPLATE.children]

        self.assertIn("物件", titles)
        self.assertLess(titles.index("地点"), titles.index("物件"))
        self.assertLess(titles.index("物件"), titles.index("人物"))
        people_node = STORY_ILLUSTRATION_TEMPLATE.children[titles.index("人物")]
        self.assertEqual(people_node.children, ())

    def test_creative_node_note_prompt_applies_medieval_battle_scope_rules(self) -> None:
        base_kwargs = {
            "project_brief": "几个对立阵营的士兵在战斗",
            "project_extra": "中世纪，田野，泥泞，傍晚",
            "project_outline": "- 几个对立阵营的士兵在战斗\n  - 世界观\n  - 时间\n  - 地点\n  - 物件\n  - 人物\n  - 事件\n  - 氛围\n  - 构图",
            "current_note": "",
            "language": "zh",
        }

        root_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="几个对立阵营的士兵在战斗",
            node_path="几个对立阵营的士兵在战斗",
        )
        world_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="世界观",
            node_path="几个对立阵营的士兵在战斗 / 世界观",
        )
        time_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="时间",
            node_path="几个对立阵营的士兵在战斗 / 时间",
        )
        place_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="地点",
            node_path="几个对立阵营的士兵在战斗 / 地点",
        )
        object_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="物件",
            node_path="几个对立阵营的士兵在战斗 / 物件",
        )
        event_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="事件",
            node_path="几个对立阵营的士兵在战斗 / 事件",
        )
        atmosphere_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="氛围",
            node_path="几个对立阵营的士兵在战斗 / 氛围",
        )
        composition_prompt = _build_creative_node_note_prompt(
            **base_kwargs,
            node_title="构图",
            node_path="几个对立阵营的士兵在战斗 / 构图",
        )

        self.assertIn("父节点/最顶层节点", root_prompt)
        self.assertIn("主题与补充信息融合后的整体画面", root_prompt)
        self.assertIn("不要体现具体事件或动作", world_prompt)
        self.assertIn("城堡、马车、木栅栏、水车", world_prompt)
        self.assertIn("傍晚", time_prompt)
        self.assertIn("不需要带世界观、地点或事件", time_prompt)
        self.assertIn("田野", place_prompt)
        self.assertIn("中世纪城堡", place_prompt)
        self.assertIn("物件/道具节点", object_prompt)
        self.assertIn("中世纪武器、泥泞马车", object_prompt)
        self.assertIn("破损工具、脚印车辙、旗帜标识", object_prompt)
        self.assertIn("search_query 必须体现事件本身", event_prompt)
        self.assertIn("多人战斗、混战、近身格斗", event_prompt)
        self.assertIn("烟雾、火花、尘土、泥水飞溅", atmosphere_prompt)
        self.assertIn("低机位、俯视、广角群像", composition_prompt)
        self.assertIn("场景空间构图、镜头位置、广角环境", composition_prompt)

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

    def test_parse_creative_project_copy_suggestion_rejects_reasoning_without_final_copy(self) -> None:
        with self.assertRaises(LLMProviderError):
            parse_creative_project_copy_suggestion(
                """
{
  "choices": [
    {
      "message": {
        "content": "",
        "reasoning": "分析：先确认节点，再检查视觉约束，但这里没有真正输出最终文案。"
      }
    }
  ]
}
                """
            )

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
        self.assertEqual(provider.calls[0]["reasoning_effort"], "none")
        self.assertEqual(provider.calls[0]["allow_reasoning_content"], True)
        messages = provider.calls[0]["messages"]
        self.assertIsInstance(messages, list)
        self.assertIn("只输出最终文案正文", messages[1]["content"])  # type: ignore[index]

    def test_generate_creative_project_copy_extracts_final_copy_from_reasoning_content(self) -> None:
        class FakeProvider(LMStudioProvider):
            def __init__(self) -> None:
                super().__init__(model_name="fake-model")

            def _chat_completion(self, **kwargs: object) -> str:  # type: ignore[override]
                assert kwargs["allow_reasoning_content"] is True
                return (
                    "分析：先检查节点约束。\n"
                    "最终文案：黄昏的自动售货机街角，飞行器驾驶员停在湿亮路面旁，"
                    "抬手按下饮料机按钮，冷蓝灯箱、透明货仓和远处低空航道共同形成轻微幽默的近未来生活场景。"
                )

        provider = FakeProvider()

        suggestion, model_name = provider.generate_creative_project_copy(
            project_brief="飞行器驾驶员买饮料",
            nodes=[{"title": "事件", "path": "项目 / 事件", "note": "买饮料动作", "search_query": ""}],
            language="zh",
        )

        self.assertEqual(model_name, "fake-model")
        self.assertTrue(suggestion.copy_text.startswith("黄昏的自动售货机街角"))
        self.assertNotIn("分析", suggestion.copy_text)

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
