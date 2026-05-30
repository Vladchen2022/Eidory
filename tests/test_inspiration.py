from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eidory.core.inspiration import InspirationTerm, mix_inspiration_search_results
from eidory.core.llm_provider import _terms_from_plain_text, parse_inspiration_proposal
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
