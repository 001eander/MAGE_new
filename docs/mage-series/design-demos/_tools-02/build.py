# -*- coding: utf-8 -*-
"""
方向 02 · 标杆迁移（Distill.pub）
从 content.json 生成 design-demos/02-benchmark-distill.html

设计逻辑：Distill 的内核是「正文是一列呼吸顺畅的文字，图和代码是正文的组成部分，
不是被装进卡片的独立模块」。迁移时保留其版式层级（窄正文、编号图注、克制的标题、
边栏索引），配色不抄 Distill，按 spec 第 7 节选 fig-paper（奶油纸底）；
「编号空间」母题落地为：左栏编号轴、首屏 256 格装置、代码片段序号。
"""
import html
import json
import os
import random
import re
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))          # .../docs/mage-series
OUT = os.path.join(DOCS, "design-demos", "02-benchmark-distill.html")

PALETTE = "fig-paper"
IMG_BASE = "../assets/" + PALETTE

with open(os.path.join(DOCS, "content.json"), encoding="utf-8") as f:
    C = json.load(f)

MONO = ('"SF Mono","Cascadia Mono","Menlo",Consolas,"DejaVu Sans Mono",'
        '"Courier New",monospace')
SERIF = ('Georgia,"Songti SC","Source Han Serif SC","Noto Serif CJK SC",'
         '"SimSun",serif')

# --------------------------------------------------------------------------
# 文本处理
# --------------------------------------------------------------------------
ALLOWED_OPEN = {"code": '<code class="ic">', "strong": "<strong>"}
TAG_RE = re.compile(r"(</?(?:code|strong)>)", re.I)


def esc(s):
    # content.json 里已有 &lt; 这类实体（如 <code>noise &lt;= cutoff</code>），
    # 先还原再统一转义，避免出现 &amp;lt;。
    return html.escape(html.unescape(s), quote=False)


def inline(s):
    """保留 <code>/<strong> 结构，只对裸文本做转义。"""
    out = []
    for part in TAG_RE.split(s):
        m = re.fullmatch(r"</?(code|strong)>", part or "", re.I)
        if m:
            name = m.group(1).lower()
            out.append("</%s>" % name if part.startswith("</") else ALLOWED_OPEN[name])
        else:
            out.append(esc(part))
    return "".join(out)


KEYWORDS = {
    "def", "class", "if", "elif", "else", "for", "while", "return", "in", "is",
    "not", "and", "or", "with", "as", "import", "from", "break", "continue",
    "try", "except", "finally", "lambda", "pass", "global", "None", "True", "False",
}
CODE_RE = re.compile(
    r"(#[^\n]*)"
    r"|('(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\")"
    r"|(\b[A-Za-z_]\w*\b)"
    r"|(\d+\.?\d*)"
)


def highlight(code):
    """给 Python 片段上色：只包 span，不改动任何字符。"""
    out, pos = [], 0
    for m in CODE_RE.finditer(code):
        out.append(esc(code[pos:m.start()]))
        gi = next(i for i in (1, 2, 3, 4) if m.group(i) is not None)
        word = m.group(0)
        if gi == 1:
            out.append('<span class="c-cm">%s</span>' % esc(word))
        elif gi == 2:
            out.append('<span class="c-st">%s</span>' % esc(word))
        elif gi == 3:
            if word in KEYWORDS:
                out.append('<span class="c-kw">%s</span>' % esc(word))
            elif word == "self":
                out.append('<span class="c-se">%s</span>' % esc(word))
            else:
                out.append(esc(word))
        else:
            out.append('<span class="c-nu">%s</span>' % esc(word))
        pos = m.end()
    out.append(esc(code[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------
# 样式（不用 CSS 变量，不用 position:fixed，无 JS，无外部请求）
# --------------------------------------------------------------------------
CSS = r"""
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:#F5F0E8;color:#241F1A;
  font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Source Han Sans SC","Noto Sans CJK SC",sans-serif;
  font-size:17px;line-height:1.85;-webkit-text-size-adjust:100%;
  overflow-wrap:break-word;word-wrap:break-word}
.page{max-width:884px;margin:0 auto;padding:62px 26px 104px}

/* ---------- 报头 ---------- */
.eyebrow{font-family:@MONO@;font-size:13px;letter-spacing:.14em;color:#8F3F22;line-height:1.7}
.eyebrow .sep{margin:0 .6em;color:#C3B69F}
h1{font-family:@SERIF@;font-size:42px;line-height:1.26;font-weight:600;
  letter-spacing:.004em;margin:22px 0 0}
.lead{margin:16px 0 0;color:#332C25}
.meta{margin:12px 0 0;font-family:@MONO@;font-size:12.5px;line-height:1.75;color:#6A6055}
.subject{margin:14px 0 0;font-size:14px;color:#4A4038}
.subject .k{font-family:@MONO@;font-size:12.5px;color:#6A6055;letter-spacing:.08em;margin-right:.7em}
.roadmap{margin:28px 0 0;padding:16px 0 0;border-top:1px solid #DED3C1;font-size:13.5px;line-height:2.1}
.rm{display:inline-block;margin:0 24px 0 0;white-space:nowrap;color:#4A4038}
.rmn{display:inline-block;min-width:1.2em;font-family:@MONO@;font-size:12.5px;color:#8F3F22}
.rml{margin-left:.35em}
.rms{margin-left:.55em;font-size:12.5px;color:#6A6055}
.rm-cur .rml{font-weight:600;color:#241F1A}

/* ---------- 首屏装置：256 格编号空间 ---------- */
.hero{margin:38px 0 0;padding:26px 0 22px;border-top:2px solid #241F1A;border-bottom:1px solid #DED3C1}
.kspace{width:256px;line-height:0;font-size:0}
.kspace i{display:inline-block;width:14px;height:14px;margin:0 2px 2px 0;
  background:#F5F0E8;border:1px solid #D6CAB4}
.kspace i.d{background:#241F1A;border-color:#241F1A}
.kspace i.m{background:#9E4526;border-color:#9E4526}
.flowline{margin:24px 0 0;font-family:@MONO@;font-size:15px;line-height:1.95;
  color:#332C25;letter-spacing:.01em}
@media (min-width:721px){
  .hero{display:flex;align-items:center;gap:34px}
  .kspace{flex:0 0 256px}
  .flowline{flex:1 1 auto;margin:0}
}

/* ---------- 目录（顶部索引，非 sticky） ---------- */
.toc{margin:46px 0 0}
.toc-h{font-family:@MONO@;font-size:13px;letter-spacing:.14em;color:#8F3F22;margin:0 0 16px}
.toc ol{list-style:none;margin:0;padding:0;border-top:1px solid #DED3C1}
.toc li{border-bottom:1px solid #EBE2D2;padding:9px 0}
.toc a{color:#241F1A;text-decoration:none;display:block;line-height:1.6}
.toc .tn{display:inline-block;width:2.6em;font-family:@MONO@;font-size:12.5px;color:#9E4526}

/* ---------- 章节：左栏编号轴 + 正文 ---------- */
main{margin-top:10px}
.sec{border-top:1px solid #DED3C1;margin-top:58px;padding-top:32px}
.sec-first{border-top:2px solid #241F1A;margin-top:56px}
.row{display:flex;align-items:baseline}
.rail{flex:0 0 72px;width:72px}
.snum{display:block;font-family:@MONO@;font-size:15px;line-height:1.2;
  color:#9E4526;letter-spacing:.02em}
.srange{display:block;font-family:@MONO@;font-size:12.5px;line-height:1.6;color:#6A6055;
  margin-top:5px;white-space:nowrap}
.scol{flex:1 1 auto;min-width:0;margin-left:22px}
h2{font-family:@SERIF@;font-size:27px;line-height:1.42;font-weight:600;
  letter-spacing:.004em;margin:0 0 22px}
h3{font-family:@SERIF@;font-size:19px;line-height:1.55;font-weight:600;color:#241F1A;
  margin:54px 0 18px;padding-top:14px;border-top:1px solid #DED3C1;letter-spacing:.01em}
h3:first-child{margin-top:0;border-top:0;padding-top:0}
h4{font-size:15px;line-height:1.65;font-weight:600;color:#4A4038;letter-spacing:.02em;
  margin:36px 0 14px;padding-top:12px;border-top:1px solid #E5DBC9}
.scol p{margin:0 0 20px;color:#2A241E}
.scol>p:last-child,.scol>ul:last-child,.scol>ol:last-child,.scol>div:last-child{margin-bottom:0}
.scol ul{list-style:none;margin:0 0 24px;padding:0}
.scol ul li{position:relative;padding-left:24px;margin:0 0 13px;color:#2A241E}
.scol ul li:before{content:"";position:absolute;left:2px;top:.72em;width:7px;height:7px;background:#9E4526}
.kp{list-style:none;margin:0;padding:0;border-top:1px solid #DED3C1}
.kp li{position:relative;margin:0;padding:17px 0 17px 46px;border-bottom:1px solid #DED3C1;color:#241F1A}
.kp .kn{position:absolute;left:0;top:18px;font-family:@MONO@;font-size:13px;
  letter-spacing:.06em;color:#9E4526}

/* ---------- 旁注 / 警示 / 提示：用尺线，不用卡片 ---------- */
.note{margin:28px 0 28px 74px;padding:16px 0;border-top:1px solid #241F1A;
  border-bottom:1px solid #241F1A;font-size:15.5px;line-height:1.85;color:#3A332B}
.warn{margin:28px 0 26px 74px;padding:18px 0 14px;border-top:2px solid #9E4526;
  border-bottom:1px solid #DED3C1;font-size:15.5px;line-height:1.85;color:#241F1A}
.tip{margin:32px 0 0;padding:14px 0 0;border-top:1px solid #DED3C1;
  font-family:@MONO@;font-size:13px;line-height:1.85;color:#6A6055}

/* ---------- 代码：浅底、无圆角、无阴影、长行横滚 ---------- */
.plate{margin:26px 0 28px -94px}
.plate-h{font-family:@MONO@;font-size:12.5px;letter-spacing:.1em;color:#6A6055;
  border-top:1px solid #DED3C1;padding-top:9px}
.plate-h b{font-weight:400;color:#9E4526}
.plate pre{margin:0;padding:18px 20px;background:#F0E9DA;overflow-x:auto;
  font-family:@MONO@;font-size:13px;line-height:1.72;color:#241F1A;white-space:pre}
.c-kw{color:#8F3F22}
.c-st{color:#5C4A2E}
.c-nu{color:#6A6055}
.c-se{color:#6A6055}
.c-cm{color:#6A6055}
.ic{font-family:@MONO@;font-size:.92em;background:#F0E9DA;color:#2A241E;padding:1px 5px}
.scol strong{font-weight:600;color:#241F1A}

/* ---------- 图：与正文同宽，图注在下方 ---------- */
.fig{margin:28px 0 34px -94px;text-align:center}
.fig a{display:block;text-decoration:none}
.fig img{width:100%;max-width:832px;height:auto;display:block;margin:0 auto}
.fig-cap{margin:13px auto 0;max-width:660px;font-size:13px;line-height:1.72;
  color:#6A6055;text-align:center}
/* 宽视口上，代码块与扁宽流程图一起向两侧出栏：
   代码最长 116 字符 ≈ 922px，出栏后 990px 可完整显示，不必横滚。 */
@media (min-width:1280px){
  .plate{margin-left:-94px;margin-right:-158px}
  .fig-wide{margin-left:-94px;margin-right:-158px}
  .fig-wide img{max-width:1084px}
}

/* ---------- 页脚 ---------- */
.foot{margin:80px 0 0;padding:22px 0 0;border-top:2px solid #241F1A;
  font-family:@MONO@;font-size:12.5px;line-height:1.9;color:#6A6055}

/* ---------- 375px 重排：单栏、不缩字号到不可读 ---------- */
@media (max-width:720px){
  .page{padding:40px 18px 72px}
  h1{font-size:30px}
  .row{display:block}
  .rail{width:auto;flex-basis:auto;margin:0 0 8px}
  .snum{display:inline;margin-right:.6em}
  .srange{display:inline;margin:0}
  .scol{margin-left:0}
  h2{font-size:23px}
  h3{font-size:17.5px;margin-top:42px}
  .plate,.fig{margin-left:0}
  .note,.warn{margin-left:0}
  .kp li{padding-left:38px}
  .plate pre{padding:14px 12px}
  .flowline{font-size:13.5px}
  .rm{display:block;margin:0 0 4px}
}
"""

STYLE = CSS.replace("@MONO@", MONO).replace("@SERIF@", SERIF)


# --------------------------------------------------------------------------
# 首屏 256 格装置
# --------------------------------------------------------------------------
def kspace_html():
    """mask_rate=0.55 的一次抽样：128 丢弃(实心墨) ⊂ 141 遮盖(锈红)；115 露出(空心)。"""
    rnd = random.Random(20260929)
    idx = list(range(256))
    rnd.shuffle(idx)
    drop = set(idx[:128])
    mask = set(idx[:141])
    cells = []
    for i in range(256):
        cls = "d" if i in drop else ("m" if i in mask else "")
        cells.append('<i class="%s"></i>' % cls if cls else "<i></i>")
    return "".join(cells)


# --------------------------------------------------------------------------
# 逐块渲染
# --------------------------------------------------------------------------
def fig_classes(index):
    """扁而宽的流程图（宽高比 < 0.35）在大视口上出栏，避免图内文字被缩得不可读。"""
    p = os.path.join(DOCS, "assets", PALETTE, "fig%d.png" % index)
    with open(p, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    return "fig fig-wide" if h / float(w) < 0.35 else "fig"


def render_block(b, frag):
    t = b["type"]
    if t == "p":
        return "<p>%s</p>" % inline(b["text"])
    if t == "flowline":
        return '<p class="flowline">%s</p>' % esc(b["text"])
    if t == "h4":
        return "<h4>%s</h4>" % esc(b["text"])
    if t == "h3":
        return "<h3>%s</h3>" % esc(b["text"])
    if t == "ul":
        return "<ul>%s</ul>" % "".join("<li>%s</li>" % inline(x) for x in b["items"])
    if t == "ol":
        return '<ol class="kp">%s</ol>' % "".join(
            '<li><span class="kn">%02d</span>%s</li>' % (i + 1, inline(x))
            for i, x in enumerate(b["items"]))
    if t == "note":
        return '<div class="note">%s</div>' % inline(b["html"])
    if t == "warnbox":
        return '<div class="warn">%s</div>' % inline(b["html"])
    if t == "tip":
        return '<div class="tip">%s</div>' % inline(b["html"])
    if t == "code":
        frag["n"] += 1
        return ('<div class="plate"><div class="plate-h">代码 <b>%02d</b></div>'
                "<pre>%s</pre></div>" % (frag["n"], highlight(b["text"])))
    if t == "figure":
        return ('<figure class="%(k)s"><a href="%(b)s/fig%(i)d.png">'
                '<img src="%(b)s/fig%(i)d.png" alt="%(c)s"></a>'
                '<figcaption class="fig-cap">%(c)s</figcaption></figure>'
                % {"k": fig_classes(b["index"]), "b": IMG_BASE,
                   "i": b["index"], "c": esc(b["caption"])})
    raise ValueError("unknown block type: %s" % t)


def build_sections():
    frag = {"n": 0}
    out = []
    for i, s in enumerate(C["sections"], 1):
        cls = "sec sec-first" if i == 1 else "sec"
        rng = ('<span class="srange">%s</span>' % esc(s["lineno"])) if s.get("lineno") else ""
        blocks = "\n".join(render_block(b, frag) for b in s["blocks"])
        out.append(
            '<section class="%s" id="sec-%d">\n<div class="row">\n'
            '<div class="rail"><span class="snum">%02d</span>%s</div>\n'
            '<div class="scol">\n<h2>%s</h2>\n%s\n</div>\n</div>\n</section>'
            % (cls, i, i, rng, esc(s["title"]), blocks))
    return "\n".join(out), frag["n"]


def build_toc():
    items = "".join(
        '<li><a href="#sec-%d"><span class="tn">%02d</span>%s</a></li>' % (i, i, esc(t))
        for i, t in enumerate(C["toc"], 1))
    return '<nav class="toc"><p class="toc-h">目录</p><ol>%s</ol></nav>' % items


def build_mast():
    s = C["series"]
    roadmap = "".join(
        '<span class="rm%s"><span class="rmn">%d</span><span class="rml">%s</span>'
        '<span class="rms">%s</span></span>'
        % (" rm-cur" if r["state"] == "本篇" else "", r["n"], esc(r["label"]), esc(r["state"]))
        for r in s["roadmap"])
    return (
        '<header class="mast">'
        '<p class="eyebrow">%s<span class="sep">/</span>第 %d 篇 / 共 %d 篇</p>'
        "<h1>%s</h1>"
        '<p class="lead">%s</p>'
        '<p class="meta">%s</p>'
        '<p class="subject"><span class="k">本篇</span>%s</p>'
        '<div class="roadmap">%s</div>'
        "</header>"
        % (esc(s["name"]), s["episode"], s["total"], esc(C["title"]),
           esc(C["lead"]), esc(C["meta"]), esc(s["subject"]), roadmap))


FLOWLINE = next(b["text"] for b in C["sections"][0]["blocks"] if b["type"] == "flowline")

DOC = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · __SERIES__ 第 __EP__ 篇</title>
<style>__CSS__</style>
</head>
<body>
<div class="page">
__MAST__
<div class="hero"><div class="kspace" aria-hidden="true">__KSPACE__</div><p class="flowline">__FLOW__</p></div>
__TOC__
<main>
__BODY__
</main>
<p class="foot">__FOOT__</p>
</div>
</body>
</html>
"""


def main():
    body, nframes = build_sections()
    doc = (DOC
           .replace("__TITLE__", esc(C["title"]))
           .replace("__SERIES__", esc(C["series"]["name"]))
           .replace("__EP__", str(C["series"]["episode"]))
           .replace("__CSS__", STYLE)
           .replace("__MAST__", build_mast())
           .replace("__KSPACE__", kspace_html())
           .replace("__FLOW__", esc(FLOWLINE))
           .replace("__TOC__", build_toc())
           .replace("__BODY__", body)
           .replace("__FOOT__", esc(C["footer"])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print("written: %s  (%d bytes, %d code fragments)" % (OUT, len(doc), nframes))


if __name__ == "__main__":
    main()
