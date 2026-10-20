# -*- coding: utf-8 -*-
import json
import os

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
HTML = os.path.join(DOCS, "design-demos", "02-benchmark-distill.html")
SHOTS = os.path.abspath(os.path.join(os.path.dirname(DOCS), "..", ".playwright-mcp"))
url = "file:///" + HTML.replace("\\", "/")
exe = os.path.join(os.environ["LOCALAPPDATA"], "ms-playwright",
                   "chromium-1234", "chrome-win64", "chrome.exe")

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(url)
    pg.wait_for_timeout(400)
    m = pg.evaluate("""() => ({
      wideFigs: [...document.querySelectorAll('.fig-wide img')].map(i => ({
        src: i.getAttribute('src').split('/').pop(),
        w: Math.round(i.getBoundingClientRect().width),
        h: Math.round(i.getBoundingClientRect().height)})),
      rail: document.querySelector('.srange').getBoundingClientRect().width,
      srangeLines: document.querySelector('.srange').getClientRects().length,
      docScrollW: document.documentElement.scrollWidth
    })""")
    print(json.dumps(m, ensure_ascii=False))
    # 滚到第 4 张图（fig4）
    pg.evaluate("() => document.querySelectorAll('.fig')[3].scrollIntoView()")
    pg.wait_for_timeout(200)
    pg.screenshot(path=os.path.join(SHOTS, "d02-1440-fig4.png"))
    pg.evaluate("() => document.querySelectorAll('.fig')[4].scrollIntoView()")
    pg.wait_for_timeout(200)
    pg.screenshot(path=os.path.join(SHOTS, "d02-1440-fig5.png"))
    b.close()
