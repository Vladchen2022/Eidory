from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreativeTemplateNode:
    title: str
    note: str
    children: tuple["CreativeTemplateNode", ...] = ()


@dataclass(frozen=True)
class CreativeTemplate:
    id: str
    label: str
    root: CreativeTemplateNode


STORY_ILLUSTRATION_TEMPLATE = CreativeTemplateNode(
    title="未命名故事性插画",
    note="用一句话概括画面中的核心事件和情绪。",
    children=(
        CreativeTemplateNode("世界观", "时代、技术水平、社会秩序或幻想规则。"),
        CreativeTemplateNode("时间", "季节、昼夜、事件发生的具体时刻。"),
        CreativeTemplateNode("地点", "场景位置、空间尺度、环境状态。"),
        CreativeTemplateNode(
            "人物",
            "参与事件的主要人物与次要人物。",
            (
                CreativeTemplateNode("主角", "画面叙事的中心人物。"),
                CreativeTemplateNode("次要角色", "推动关系或冲突的人物。"),
            ),
        ),
        CreativeTemplateNode("事件", "画面中正在发生的动作、冲突或转折。"),
        CreativeTemplateNode("氛围", "情绪、光线、色彩倾向和观看感受。"),
        CreativeTemplateNode("构图", "视角、主体位置、画面层次和视觉焦点。"),
    ),
)

SCENE_INTERIOR_TEMPLATE = CreativeTemplateNode(
    title="未命名室内人文环境",
    note="概括室内空间的功能、使用者、时代痕迹和最强视觉记忆点。",
    children=(
        CreativeTemplateNode("世界观", "空间所属时代、文明、职业系统、技术水平或宗教/权力结构。"),
        CreativeTemplateNode("空间功能", "这里被谁使用，用来完成什么活动，公共、私人或半封闭属性。"),
        CreativeTemplateNode("建筑与布局", "房间尺度、层高、入口、窗、隔断、楼梯、走廊和视线遮挡。"),
        CreativeTemplateNode("动线与视角", "人物进入、停留、转身、观察的路径，以及画面主视角。"),
        CreativeTemplateNode("陈设与道具", "家具、器械、容器、屏幕、工具、书册、餐具或工作台等可见物。"),
        CreativeTemplateNode("材质表面", "墙面、地面、织物、木材、金属、玻璃、灰尘、油污、划痕和修补。"),
        CreativeTemplateNode("灯光与色彩", "窗光、灯具、屏幕光、火光等实际光源，以及主色、辅助色和暗部颜色。"),
        CreativeTemplateNode("生活痕迹", "刚离开的人、未完成的工作、翻倒物、脚印、水渍、张贴物或遗留衣物。"),
        CreativeTemplateNode("构图焦点", "画面中心物、前中后景层次、遮挡关系、留白和引导线。"),
    ),
)

SCENE_EXTERIOR_TEMPLATE = CreativeTemplateNode(
    title="未命名室外人文环境",
    note="概括室外人工环境的地点类型、社会功能、建筑轮廓和街景叙事。",
    children=(
        CreativeTemplateNode("世界观", "城市或聚落所属时代、制度、科技、能源、审美和生活方式。"),
        CreativeTemplateNode("地点类型", "街道、广场、港口、市场、车站、工业区、边境、废墟或临时营地。"),
        CreativeTemplateNode("地形与尺度", "道路宽度、高差、坡道、桥梁、台阶、建筑高度和远景边界。"),
        CreativeTemplateNode("建筑群轮廓", "建筑风格、屋顶线、门窗节奏、立面材料、招牌和附加结构。"),
        CreativeTemplateNode("交通与动线", "车辆、人流、货物流、门口排队、路障、轨道、停靠点和危险区域。"),
        CreativeTemplateNode("公共设施", "路灯、管线、广告牌、摊位、监控、座椅、围栏、垃圾桶和维修口。"),
        CreativeTemplateNode("人群活动痕迹", "摊贩摆放、排队路线、涂鸦、脚印、积水、临时遮棚和破损修补。"),
        CreativeTemplateNode("天气与光照", "日照方向、阴影长度、雨雪雾尘、霓虹反光、傍晚天光或夜间灯源。"),
        CreativeTemplateNode("构图焦点", "主建筑、交叉路口、远处地标、前景遮挡和视线引导线。"),
    ),
)

SCENE_NATURAL_TEMPLATE = CreativeTemplateNode(
    title="未命名自然环境",
    note="概括自然场景的地貌、生态、气候、尺度和画面中的路径或焦点。",
    children=(
        CreativeTemplateNode("世界观", "自然环境是否现实、幻想、异星、灾后或被某种力量改变。"),
        CreativeTemplateNode("地貌结构", "山体、峡谷、海岸、洞穴、森林、湿地、沙漠、冰原或火山的空间骨架。"),
        CreativeTemplateNode("植被生态", "树冠形状、草丛密度、藤蔓、苔藓、花期、枯枝和植物分层。"),
        CreativeTemplateNode("水体与气候", "河流、瀑布、潮汐、积雪、雾、雨、风向、云层和空气湿度。"),
        CreativeTemplateNode("岩石土壤材质", "岩层纹理、泥土颜色、砂砾颗粒、湿滑表面、裂缝、冰面或火山灰。"),
        CreativeTemplateNode("生命迹象", "动物足迹、巢穴、羽毛、骨骼、昆虫群、被啃咬植物或隐藏生物。"),
        CreativeTemplateNode("尺度参照", "人物、树木、巨石、瀑布、远山、飞鸟或建筑残片用来显示巨大或狭小。"),
        CreativeTemplateNode("时间与光照", "清晨、正午、黄昏、月夜、逆光、斑驳树影、云隙光或强烈反射。"),
        CreativeTemplateNode("路径与危险", "可行走路线、断崖、沼泽、落石、隐藏入口、迷失区域或安全落脚点。"),
        CreativeTemplateNode("构图焦点", "最高点、最亮区域、路径尽头、洞口、独特植物或异常自然现象。"),
    ),
)

CHARACTER_DESIGN_TEMPLATE = CreativeTemplateNode(
    title="未命名角色设计",
    note="概括角色身份、所属世界、身体识别点和最强视觉记忆点。",
    children=(
        CreativeTemplateNode("世界观", "角色所属时代、种族/文明、技术或魔法规则、社会阶层和环境压力。"),
        CreativeTemplateNode("身份", "职业、阵营、社会位置、日常职责和画面中能看见的身份标记。"),
        CreativeTemplateNode("身体结构", "身高比例、体型、骨架、肌肉/机械/异形结构、姿态重心和运动方式。"),
        CreativeTemplateNode("头面部", "脸型、五官、发型、年龄感、表情、伤痕、妆容、义眼或其他识别点。"),
        CreativeTemplateNode("穿戴", "服装剪裁、层次、材质、颜色、护具、鞋靴、磨损、污渍和文化来源。"),
        CreativeTemplateNode("物件", "随身工具、武器、包袋、饰品、证件、维修痕迹或和身份绑定的特殊物。"),
    ),
)

OBJECT_DESIGN_TEMPLATE = CreativeTemplateNode(
    title="未命名物件设计",
    note="概括物件用途、所属世界和最强视觉识别点。",
    children=(
        CreativeTemplateNode("世界观", "物件所属时代、技术体系、制造文化、使用环境和审美来源。"),
        CreativeTemplateNode("使用者与用途", "谁使用它，解决什么问题，单手/双手/多人/固定安装的使用方式。"),
        CreativeTemplateNode("整体轮廓", "第一眼的剪影、比例、重心、可握持位置、展开或收纳后的外形。"),
        CreativeTemplateNode("结构拆解", "主体、接口、关节、开合件、按钮、管线、容器、能源仓和连接方式。"),
        CreativeTemplateNode("材质工艺", "金属、塑料、木材、皮革、陶瓷、玻璃、织物、铸造、焊接或手工痕迹。"),
        CreativeTemplateNode("交互细节", "屏幕、刻度、指示灯、拉环、旋钮、锁扣、磨砂握把和操作反馈。"),
        CreativeTemplateNode("使用痕迹", "磨损、污渍、维修补丁、贴纸、铭牌、刮痕、掉漆、裂纹和临时改装。"),
        CreativeTemplateNode("工作状态", "静止、启动、过热、损坏、展开、装填、放电、泄漏或被拆解的状态。"),
        CreativeTemplateNode("展示尺度", "与手、人物、桌面、载具或建筑构件的比例关系。"),
    ),
)

CREATIVE_TEMPLATES: tuple[CreativeTemplate, ...] = (
    CreativeTemplate("story", "故事性插画", STORY_ILLUSTRATION_TEMPLATE),
    CreativeTemplate("sceneInterior", "场景设计：室内人文环境", SCENE_INTERIOR_TEMPLATE),
    CreativeTemplate("sceneExterior", "场景设计：室外人文环境", SCENE_EXTERIOR_TEMPLATE),
    CreativeTemplate("sceneNatural", "场景设计：自然环境", SCENE_NATURAL_TEMPLATE),
    CreativeTemplate("character", "角色设计", CHARACTER_DESIGN_TEMPLATE),
    CreativeTemplate("object", "物件设计", OBJECT_DESIGN_TEMPLATE),
)


def creative_template_by_id(template_id: str) -> CreativeTemplate:
    for template in CREATIVE_TEMPLATES:
        if template.id == template_id:
            return template
    return CREATIVE_TEMPLATES[0]


def template_search_query(title: str, note: str, project_brief: str = "") -> str:
    parts = [project_brief.strip(), title.strip(), note.strip()]
    return " ".join(part for part in parts if part)[:400]
