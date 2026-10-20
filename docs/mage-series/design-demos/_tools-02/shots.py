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

MEASURE = """() => {
  const vw = document.documentElement.clientWidth;
  const pres = [...document.querySelectorAll('pre')];
  return {
    vw, scrollW: document.documentElement.scrollWidth,
    pageW: Math.round(document.querySelector('.page').getBoundingClientRect().width),
    scolW: Math.round(document.querySelector('.scol').getBoundingClientRect().width),
    plateW: Math.round(document.querySelector('.plate').getBoundingClientRect().width),
    plateLeft: Math.round(document.querySelector('.plate').getBoundingClientRect().left),
    clippedPres: pres.filter(p => p.scrollWidth > p.clientWidth + 1)
                      .map(p => p.scrollWidth - p.clientWidth),
    figW: [...document.querySelectorAll('.fig img')].map(i => Math.round(i.getBoundingClientRect().width)),
    kspace: [Math.round(document.querySelector('.kspace').getBoundingClientRect().width),
             Math.round(document.querySelector('.kspace').getBoundingClientRect().height)],
    bodyFont: getComputedStyle(document.body).fontSize,
    offscreen: [...document.querySelectorAll('body *')].filter(e => {
      const c = e.closest('pre'); return !c && e.getBoundingClientRect().right > vw + 1;
    }).length,
    brokenImgs: [...document.querySelectorAll('img')].filter(i => !i.naturalWidth).length,
  };
}"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe)
    for w, h, tag in [(1440, 900, "1440"), (1024, 800, "1024"), (768, 900, "768"), (375, 812, "375")]:
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto(url)
        pg.wait_for_timeout(350)
        print(tag, json.dumps(pg.evaluate(MEASURE), ensure_ascii=False))
        if tag in ("1440", "375"):
            pg.evaluate("() => window.scrollTo(0,0)")
            pg.screenshot(path=os.path.join(SHOTS, "d02-%s-top.png" % tag))
            pg.evaluate("() => document.getElementById('sec-10').scrollIntoView()")
            pg.wait_for_timeout(150)
            pg.screenshot(path=os.path.join(SHOTS, "d02-%s-sec10.png" % tag))
        pg.close()
    b.close()
