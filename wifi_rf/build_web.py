"""把 app_data.json 注入 web/index.html，产出单文件交付版。"""
from __future__ import annotations

import json
import os

from . import config as C

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
DST = os.path.join(C.OUT_DIR, "wifi-coverage-studio.html")


def build() -> str:
    with open(SRC, encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(C.OUT_DIR, "app_data.json"), encoding="utf-8") as f:
        data = json.load(f)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    assert "/*__DATA__*/" in html
    html = html.replace("/*__DATA__*/", payload)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(html)
    print("->", DST, round(os.path.getsize(DST) / 1024), "KB")
    return DST


if __name__ == "__main__":
    build()
