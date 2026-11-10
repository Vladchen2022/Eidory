"""Shared language for decisions that change a library image."""

REMOVE_FROM_LIBRARY = "从图库移除"
TRASH_SOURCE = "移到废纸篓"


def describe_review_actions(copy, replace, discard, *, copy_is_generated=False):
    copy.setText("保存到图库…" if copy_is_generated else "另存副本…")
    copy.setToolTip("选择保存位置；保存后加入图库。" +
                    ("" if copy_is_generated else "原图保持不变。"))
    copy.setProperty("variant", "primary")
    replace.setText("替换原图")
    replace.setToolTip("用结果更新原图的文件位置；可从原图备份恢复。")
    discard.setText("放弃结果")
    discard.setToolTip("不采用此次结果；原图保持不变。")
