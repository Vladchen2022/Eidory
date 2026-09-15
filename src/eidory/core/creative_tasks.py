"""Node responsibilities determine the task, independently of its display format."""
from eidory.core.creative_templates import creative_node_template


def node_task(title, path=''):
    template = creative_node_template(title, path)
    name = template.title if template else title
    if name in {'人物', '主角', '次要角色'}:
        return ('完善人物方案', 'Develop Characters',
                '按已确定的人物分别补充分工、身份差异、服装轮廓、动作和表情。人数已确定时严禁加人；'
                '每行对应一位已有人物或一个明确的人物造型方向，name 是可独立找图的人物身份，visual 是可见区别。'
                '不要用工具带、腕部终端等配件代替人物方案。未确定人数时提出少量候选角色，明确是候选而非新增事实。')
    if name in {'地点', '建筑与布局', '地点类型', '地形与尺度', '建筑群轮廓', '地貌结构'}:
        return ('补充空间组成', 'Develop Space',
                '补充符合此节点的空间结构、布局单元或自然地貌。每行是可单独找图的空间目标，说明其尺度、相互位置与用途。'
                '不要用散落的小道具代替空间布局；保留用户已确定的场所。')
    if name == '穿戴':
        return ('完善穿戴方案', 'Develop Clothing',
                '为已有人物补充服装层次、剪裁、护具或携带装备。每行是独立的穿戴参考目标，服从身份与文化，不改变人数或身体。')
    if name in {'结构拆解', '交互细节', '使用痕迹'}:
        return ('补充设计要素', 'Develop Design Elements',
                '围绕既定物件的结构、交互或使用痕迹补充可见要素，不另造新的产品。每行必须属于当前节点职责，不能编造工程可行性。')
    if template and template.content_format == 'text':
        return ('完善说明', 'Develop Description', '按节点职责完善一段连贯说明，保留所有已确定内容。')
    return ('联想新物件', 'Suggest Objects', '')


def role_contract(title, path, language='zh'):
    label, _, instruction = node_task(title, path)
    if not instruction or label == '完善说明':
        return ''
    return ('Task: ' + label + '. ' + instruction + '\n'
            'Follow this node responsibility instead of a generic new-prop inventory. '
            'Preserve the brief and confirmed node content. Return only new compatible contributions in additions '
            '[{name: searchable target, visual: visible design and use}], plus search_query. '
            'Do not repeat existing descriptions. Empty additions are allowed. '
            'Keep confirmed counts, identities, time and world rules unchanged. '
            + ('Write all output fields in English.' if language == 'en' else '所有输出字段使用简体中文。'))
