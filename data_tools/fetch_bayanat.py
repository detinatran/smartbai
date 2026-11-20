"""smartbai: download UAE industrial/manufacturing datasets from bayanat.ae (UAE federal open data portal).
For each dataset matched by the searches, every resource is fetched in full through the portal's public API
(GetDatasetResource, GUID taken from the resource's visualization page) and saved as JSON and CSV.
Resumable: datasets whose folder already has index.json are skipped. usage: fetch_bayanat.py OUTDIR"""
import csv
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bayanat")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
BASE = "https://bayanat.ae"
QUERIES = ("industrial", "manufacturing")


def get(url, tries=3, timeout=40):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # retry, then raise
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:90]


def save_rows(rows, stem):
    stem.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    if rows and isinstance(rows, list) and isinstance(rows[0], dict):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


summary = []
seen = set()
for q in QUERIES:
    for page in (1, 2):
        s = get(f"{BASE}/en/Datasets?search={q}&page={page}").decode("utf-8", "ignore")
        for m in re.finditer(r'href="(/en/Datasets/Dataset-info\?id=[^"]+)"[^>]*>\s*(?:<[^>]+>\s*)*([^<]+)', s):
            title = html.unescape(m.group(2)).strip()
            if not title or title in seen:
                continue
            seen.add(title)
            folder = OUT / slug(title)
            if (folder / "index.json").exists():
                summary.append(json.loads((folder / "index.json").read_text(encoding="utf-8")))
                print("skip (done):", title, flush=True)
                continue
            folder.mkdir(exist_ok=True)
            rec = dict(title=title, query=q, source=BASE + m.group(1).split("?")[0], resources=[])
            try:
                d = get(BASE + m.group(1)).decode("utf-8", "ignore")
                pub = re.search(r"(Ministry Of Economy|Federal Competitiveness and Statistics Cent\w+|Statistics Centre\s*\S\s*Abu Dhabi|General Civil Aviation Authority)", d)
                rec["publisher"] = pub.group(1) if pub else None
                upd = re.search(r"Last Updated Date\s*(?:<[^>]+>\s*)*([0-9/]+)", d)
                rec["last_updated"] = upd.group(1) if upd else None
                items = [json.loads(html.unescape(x)) for x in re.findall(r"data-item='(\{.*?\})'", d)]
                vizs = [html.unescape(v) for v in re.findall(r'href="(Dataset-info/visualization\?id=[^"]+)"', d)]
                for i, it in enumerate(items):
                    name = (it.get("NameEN") or f"resource_{i}").strip(" .")
                    r = dict(name=name, original_file_url=it.get("FileURL"), status=None)
                    viz = next((v for v in vizs if it.get("ID", "~") in v), None)
                    guid = None
                    if viz:
                        vh = get(f"{BASE}/en/Datasets/{viz}").decode("utf-8", "ignore")
                        g = re.search(r"GetDatasetResource\?resourceID=([0-9a-f-]{36})", vh)
                        guid = g.group(1) if g else None
                    if guid:
                        rows = json.loads(get(f"{BASE}/api/DatasetResources/GetDatasetResource?resourceID={guid}") or b"null")
                        save_rows(rows, folder / slug(name))
                        r.update(status="ok", guid=guid, rows=len(rows) if isinstance(rows, list) else None)
                    else:
                        prev = (it.get("DataResourcePreview") or {}).get("Text")
                        if prev:
                            save_rows(json.loads(prev), folder / (slug(name) + ".preview"))
                        r["status"] = "preview_only (no API GUID)"
                    rec["resources"].append(r)
                (folder / "index.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            summary.append(rec)
            print(title, "|", [(x["status"], x.get("rows")) for x in rec["resources"]], rec.get("error", ""), flush=True)
            time.sleep(1)
(OUT / "index.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print("datasets:", len(summary), "errors:", sum("error" in r for r in summary))
