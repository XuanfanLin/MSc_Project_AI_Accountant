import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
URL_LIST_FILE = "urls/legislation_urls.txt"
SAVE_PATH = "uk_tax_law_extraction/legislation_data.json"

def clean_text(text):
    return ' '.join(text.strip().split())

def extract_section_title(soup):
    heading = soup.find("h1")
    if heading:
        return heading.get_text(strip=True)
    return "Unknown Title"

def extract_section_number(url):
    match = re.search(r'/section/(\d+)', url)
    return f"Section {match.group(1)}" if match else "Unknown Section"

def extract_legislation_body(soup):
    law_text = []
    for tag in soup.select("div#content p, div#content li"):
        text = clean_text(tag.get_text())
        # 跳过冗余法令更改说明、重复内容
        if ("Changes to legislation" in text or
            "Revised legislation carried on this site" in text or
            "Back to top" in text or
            "Textual Amendments" in text):
            continue
        if len(text) > 10:
            law_text.append(text)

    # 去重（避免重复条款被堆叠）
    seen = set()
    unique_text = []
    for line in law_text:
        if line not in seen:
            unique_text.append(line)
            seen.add(line)

    return " ".join(unique_text)

def fetch_legislation_page(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.content, "html.parser")

        section = extract_section_number(url)
        title = f"{section} – {extract_section_title(soup)}"
        content = extract_legislation_body(soup)

        if len(content) < 100:
            return None

        return {
            "url": url,
            "title": title,
            "content": content
        }

    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
        return None

def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    with open(URL_LIST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    results = []
    for idx, url in enumerate(urls):
        print(f"[{idx+1}/{len(urls)}] Fetching {url}")
        data = fetch_legislation_page(url)
        if data:
            results.append(data)
        else:
            print(f"[SKIP] No valid legislation found at {url}")
        time.sleep(0.5)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} entries to {SAVE_PATH}")

if __name__ == "__main__":
    main()