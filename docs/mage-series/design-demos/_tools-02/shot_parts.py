# -*- coding: utf-8 -*-
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
    for sel, name in [(".warn", "warn"), (".note", "note"), (".kp", "kp")]:
        pg.evaluate("s => document.querySelector(s).scrollIntoView({block:'center'})", sel)
        pg.wait_for_timeout(200)
        pg.screenshot(path=os.path.join(SHOTS, "d02-1440-%s.png" % name))
    b.close()
