# -*- coding: utf-8 -*-
"""校验 02-benchmark-distill.html 与 content.json 的对齐度。"""
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
HTML_PATH = os.path.join(DOCS, "design-demos", "02-benchmark-distill.html")

with open(os.path.join(DOCS, "content.json"), encoding="utf-8") as f:
    C = json.load(f)
raw = open(HTML_PATH, encoding="utf-8").read()

# 剥掉 <style>，其余标签剥掉，实体还原，空白归一，得到"可见文本"
body = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
text = re.sub(r"<[^>]+>", "", body)
text = html.unescape(text)
text = re.sub(r"\s+", " ", text)

fails = []


def need(label, s):
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()
    if t and t not in text:
        fails.append("%s: %s" % (label, t[:70]))


# ---- 1. 统计类 ----
n_sec_doc = raw.count('<section class="sec')
n_sec_json = len(C["sections"])
n_toc_doc = len(re.findall(r'<li><a href="#sec-\d+">', raw))
n_code_doc = len(re.findall(r'<div class="plate">', raw))
n_code_json = sum(1 for s in C["sections"] for b in s["blocks"] if b["type"] == "code")
n_fig_doc = len(re.findall(r'<figure class="fig[^"]*">', raw))
n_fig_json = sum(1 for s in C["sections"] for b in s["blocks"] if b["type"] == "figure")
n_img_doc = len(re.findall(r'<img src="\.\./assets/fig-paper/fig\d\.png"', raw))
n_toc_json = len(C["toc"])

print("== 统计 ==")
for name, doc, js in [("section", n_sec_doc, n_sec_json),
                      ("toc 条目", n_toc_doc, n_toc_json),
                      ("代码块", n_code_doc, n_code_json),
                      ("figure", n_fig_doc, n_fig_json),
                      ("img(相对路径)", n_img_doc, n_fig_json)]:
    ok = "OK " if doc == js else "!! "
    print("%s%-14s html=%-3d content.json=%-3d" % (ok, name, doc, js))
    if doc != js:
        fails.append("count %s: %d != %d" % (name, doc, js))

print("== 内容完整性 ==")
need("title", C["title"])
need("lead", C["lead"])
need("meta", C["meta"])
need("series.name", C["series"]["name"])
for r in C["series"]["roadmap"]:
    need("roadmap#%d.label" % r["n"], r["label"])
    need("roadmap#%d.state" % r["n"], r["state"])
need("series.subject", C["series"]["subject"])
for i, t in enumerate(C["toc"], 1):
    need("toc[%d]" % i, t)
for i, s in enumerate(C["sections"], 1):
    need("sec%d.title" % i, s["title"])
    if s.get("lineno"):
        need("sec%d.lineno" % i, s["lineno"])
    for j, b in enumerate(s["blocks"]):
        tag = "sec%d.block%d(%s)" % (i, j, b["type"])
        if b["type"] in ("p", "flowline", "h3", "h4"):
            need(tag, b["text"])
        elif b["type"] in ("ul", "ol"):
            for k, it in enumerate(b["items"]):
                need(tag + ".item%d" % k, it)
        elif b["type"] in ("note", "warnbox", "tip"):
            need(tag, b["html"])
        elif b["type"] == "code":
            need(tag, b["text"])
        elif b["type"] == "figure":
            need(tag + ".caption", b["caption"])
need("footer", C["footer"])

# 系列识别
m = re.search(r"第\s*%d\s*篇\s*/\s*共\s*%d\s*篇" % (C["series"]["episode"], C["series"]["total"]), text)
print("%s系列标记 第 %d 篇 / 共 %d 篇" % ("OK " if m else "!! ", C["series"]["episode"], C["series"]["total"]))
if not m:
    fails.append("series marker missing")

# 代码保真：剥标签后必须与原文逐字相等
print("== 代码保真（去标签后逐字比对）==")
blocks = re.findall(r"<pre>(.*?)</pre>", raw, flags=re.S)
codes = [b["text"] for s in C["sections"] for b in s["blocks"] if b["type"] == "code"]
if len(blocks) != len(codes):
    fails.append("pre count %d != code count %d" % (len(blocks), len(codes)))
bad = 0
for i, (got, want) in enumerate(zip(blocks, codes), 1):
    plain = html.unescape(re.sub(r"<[^>]+>", "", got))
    if plain != want:
        bad += 1
        fails.append("code#%d 不保真" % i)
print("%s23 段代码逐字一致：%d/%d" % ("OK " if bad == 0 else "!! ", len(codes) - bad, len(codes)))

# 反 slop / 红线自检
print("== 红线自检 ==")
checks = [
    ("无 <script>", "<script" not in raw.lower()),
    ("无 CSS var()", "var(" not in raw),
    ("无 position:fixed", "position:fixed" not in raw.replace(" ", "")),
    ("无外部请求(http)", "http://" not in raw and "https://" not in raw),
    ("无 border-radius", "border-radius" not in raw),
    ("无 box-shadow", "box-shadow" not in raw),
    ("无 emoji 图标", not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", raw)),
    ("无「以下略/占位」", not re.search(r"以下略|TODO|占位符|\.\.\.\.\.\.", raw)),
    ("字体栈无 Inter/Roboto/Arial", not re.search(r"\bInter\b|\bRoboto\b|\bArial\b", raw)),
    ("无渐变", "gradient" not in raw.lower()),
]
for name, ok in checks:
    print("%s%s" % ("OK " if ok else "!! ", name))
    if not ok:
        fails.append(name)

# 尺寸关键值
print("== 尺寸 ==")
for pat in [r"max-width:884px", r"font-size:17px", r"font-size:13px", r"font-size:12\.5px",
            r"@media \(max-width:720px\)"]:
    ok = re.search(pat, raw) is not None
    print("%s%s" % ("OK " if ok else "!! ", pat))
    if not ok:
        fails.append("css missing " + pat)

print()
if fails:
    print("!! 失败 %d 项：" % len(fails))
    for x in fails:
        print("   -", x)
    sys.exit(1)
print("全部校验通过。")
