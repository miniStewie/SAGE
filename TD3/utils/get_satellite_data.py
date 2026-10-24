import json
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE = "https://spacemapper.cn"
SEARCH_URL = f"{BASE}/satellite/orbital/search"


from typing import Any, Optional, Dict

def _is_orbit_dict(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and ("LINE1" in x)
        and ("LINE2" in x)
        and ("OBJECT_NAME" in x or "NORAD_CAT_ID" in x)
    )

def normalize_orbit(obj: Any) -> Optional[Dict]:
    """
    把各种形态统一成“根数 dict”：
    - 直接是根数对象：{OBJECT_NAME, LINE1, LINE2, ...}
    - 包了一层：{"orbit": {...}} / {"data": {...}} / {"result": {...}}
    - 列表：[{...}, ...]
    - 其他：返回 None
    """
    if obj is None:
        return None

    if _is_orbit_dict(obj):
        return obj

    if isinstance(obj, dict):
        # 常见包裹层
        for key in ("orbit", "data", "result", "payload"):
            inner = obj.get(key)
            if _is_orbit_dict(inner):
                return inner

        # 兜底：遍历值，找第一个像根数的 dict
        for v in obj.values():
            got = normalize_orbit(v)
            if got:
                return got

    if isinstance(obj, list):
        for it in obj:
            got = normalize_orbit(it)
            if got:
                return got

    return None


@dataclass(frozen=True)
class SatMeta:
    cat_id: int
    norad_id: int
    name: str
    channel: str
    details_url: str


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s


def extract_details_links(html: str) -> List[SatMeta]:
    """
    从列表页 HTML 中提取详情页链接（/satellite/orbital/details?...）
    """
    soup = BeautifulSoup(html, "lxml")
    metas: Dict[int, SatMeta] = {}

    for a in soup.select('a[href*="/satellite/orbital/details?"]'):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        abs_url = urljoin(BASE, href)
        q = parse_qs(urlparse(abs_url).query)

        if "catId" not in q:
            continue

        try:
            cat_id = int(q["catId"][0])
        except Exception:
            continue

        channel = q.get("channel", ["KY"])[0] or "KY"
        name = unquote(q.get("name", [""])[0])
        norad = q.get("noradId", q.get("catId", ["0"]))[0]
        try:
            norad_id = int(norad)
        except Exception:
            norad_id = cat_id

        if not name:
            continue

        metas.setdefault(
            cat_id,
            SatMeta(
                cat_id=cat_id,
                norad_id=norad_id,
                name=name,
                channel=channel,
                details_url=abs_url,
            ),
        )

    return list(metas.values())


def collect_first_n_sats(
    sess: requests.Session,
    constellation_id: int,
    channel: str,
    n: int,
    page_size: int = 20,
) -> List[SatMeta]:
    results: List[SatMeta] = []
    seen = set()

    page_index = 1
    while len(results) < n:
        params = {
            "constellationId": constellation_id,
            "pageIndex": page_index,
            "pageSize": page_size,
            "channel": channel,
        }
        r = sess.get(SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()

        metas = extract_details_links(r.text)

        # 防止列表页是 JS 渲染导致抓不到
        if not metas and page_index == 1:
            raise RuntimeError("列表页 HTML 未解析到详情链接：可能该页是 JS 渲染，需要改用 Playwright 抓列表页。")

        for m in metas:
            if m.cat_id in seen:
                continue
            seen.add(m.cat_id)
            results.append(m)
            if len(results) >= n:
                break

        page_index += 1
        time.sleep(random.uniform(0.2, 0.6))

        if not metas:
            break

    return results[:n]


def try_extract_orbit_from_html(html: str) -> Optional[Dict]:
    """
    直接从 HTML 提取 #pane-JSON 下 <pre><code> 的 JSON。
    成功返回 dict；失败返回 None
    """
    soup = BeautifulSoup(html, "lxml")

    # 1) 结构化抓取（最干净）
    code_nodes = soup.select("#pane-JSON pre code")
    for node in code_nodes:
        txt = (node.get_text() or "").strip()
        if '"OBJECT_NAME"' in txt and '"LINE1"' in txt and '"LINE2"' in txt:
            txt2 = extract_json_object(txt)
            if txt2:
                return json.loads(txt2)

    # 2) 兜底：整页正则找一段包含关键字段的 JSON
    # 注意：你这个根数 JSON 是“扁平对象”，一般不会嵌套大括号，所以这样够用
    m = re.search(r'\{[^{}]*"OBJECT_NAME"[^{}]*"LINE1"[^{}]*"LINE2"[^{}]*\}', html, re.S)
    if m:
        txt2 = extract_json_object(m.group(0))
        if txt2:
            return json.loads(txt2)

    return None


def extract_json_object(text: str) -> Optional[str]:
    """
    从一段文本里截取最外层 {...}，用于容错（避免前后有多余字符）。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1].strip()
    return None


def fetch_orbit_via_playwright(details_url: str, timeout_ms: int = 45000, headless: bool = True, retries: int = 3) -> Optional[Dict]:
    """
    Playwright 打开详情页，从 DOM 中抓 #pane-JSON pre code 的文本。
    - 自动重试 retries 次
    - 失败返回 None（不抛异常）
    """
    from playwright.sync_api import sync_playwright

    def looks_like_orbit_json(txt: str) -> bool:
        return ('"OBJECT_NAME"' in txt) and ('"LINE1"' in txt) and ('"LINE2"' in txt)

    for attempt in range(1, retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                page = browser.new_page()
                page.goto(details_url, wait_until="domcontentloaded", timeout=timeout_ms)

                # 点 JSON tab（更稳）
                try:
                    page.get_by_role("tab", name=re.compile(r"^\s*JSON\s*$")).click(timeout=timeout_ms)
                except Exception:
                    try:
                        page.get_by_text(re.compile(r"^\s*JSON\s*$")).first.click(timeout=timeout_ms)
                    except Exception:
                        pass

                # 等待 code 出现（可能需要一点时间）
                locator = page.locator("#pane-JSON .code-block code").first
                locator.wait_for(state="attached", timeout=timeout_ms)

                # 多等一会儿让内容填充
                page.wait_for_timeout(800 + 400 * attempt)

                txt = locator.inner_text(timeout=timeout_ms).strip()
                browser.close()

            if not txt:
                continue
            if not looks_like_orbit_json(txt):
                continue

            txt2 = extract_json_object(txt)
            if not txt2:
                continue

            return json.loads(txt2)

        except Exception:
            # 任何异常都进入下一次重试
            time.sleep(0.6 * attempt)

    return None



def main():
    # 你固定的参数：constellationId=104, channel=KY
    constellation_id = 104
    channel = "KY"
    n = 50
    out = "spacemapper_orbits_50.json"

    sess = build_session()
    sats = collect_first_n_sats(sess, constellation_id, channel, n)

    data = []
    failed = []

    for meta in tqdm(sats, desc="Saving 50 orbit JSON"):
        orbit = None

        # 先 requests 解析 HTML
        try:
            r = sess.get(meta.details_url, timeout=30)
            r.raise_for_status()
            orbit = try_extract_orbit_from_html(r.text)
        except Exception:
            orbit = None

        # HTML 没有再 Playwright
        if orbit is None:
            orbit = fetch_orbit_via_playwright(meta.details_url, headless=True, retries=3)

        # 仍失败：跳过，并记录
        if orbit is None:
            failed.append({
                "catId": meta.cat_id,
                "noradId": meta.norad_id,
                "name": meta.name,
                "channel": meta.channel,
                "detailsUrl": meta.details_url,
                "reason": "orbit_json_not_found",
            })
            continue

        orbit = normalize_orbit(orbit)

        if orbit is None:
            failed.append({
                "catId": meta.cat_id,
                "noradId": meta.norad_id,
                "name": meta.name,
                "detailsUrl": meta.details_url,
                "reason": "orbit_normalize_failed",
            })
            continue

        # 关键：只保存根数对象本体
        data.append(orbit)

        time.sleep(random.uniform(0.2, 0.6))

    # 保存成功与失败
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


    print(f"Done. success={len(data)}, failed={len(failed)} -> {out}, failed.json")


if __name__ == "__main__":
    main()
