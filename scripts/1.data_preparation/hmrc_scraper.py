import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
URL_LIST_FILE = "urls/hmrc_urls.txt"
SAVE_PATH = "uk_tax_law_extraction/hmrc_data.json"

def clean_text(text):
    return ' '.join(text.strip().split())

def extract_manual_code(url):
    match = re.search(r'/([a-z]{4}\d{4,5})$', url)
    return match.group(1).upper() if match else "HMRC Manual"

def fetch_hmrc_page(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.content, "html.parser")

        title_tag = soup.find("h1")
        manual_code = extract_manual_code(url)
        title = f"{manual_code} – {title_tag.get_text(strip=True) if title_tag else 'Untitled'}"

        content_blocks = soup.select("div.gem-c-govspeak, .govuk-govspeak")
        paragraphs = []

        for block in content_blocks:
            for tag in block.find_all(["p", "li", "h2", "h3", "div"]):
                text = tag.get_text(separator=' ', strip=True)
                if text and "Freedom of Information Act" not in text:
                    paragraphs.append(clean_text(text))

        full_text = " ".join(paragraphs)
        if len(full_text) < 100:
            return None

        return {
            "url": url,
            "title": title,
            "content": full_text
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
        data = fetch_hmrc_page(url)
        if data:
            results.append(data)
        else:
            print(f"[SKIP] No valid content found at {url}")
        time.sleep(0.5)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} entries to {SAVE_PATH}")

if __name__ == "__main__":
    main()