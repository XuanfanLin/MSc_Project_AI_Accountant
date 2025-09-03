import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

# ==== Config ====
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
URL_LIST_FILE = "urls/hmrc_urls.txt"              # File containing list of HMRC URLs
SAVE_PATH = "uk_tax_law_extraction/hmrc_data.json"  # Output JSON file to save results

def clean_text(text):
    """Normalize whitespace in extracted text."""
    return ' '.join(text.strip().split())

def extract_manual_code(url):
    """Extract HMRC manual code (e.g., EIM32760) from URL if available."""
    match = re.search(r'/([a-z]{4}\d{4,5})$', url)
    return match.group(1).upper() if match else "HMRC Manual"

def fetch_hmrc_page(url):
    """Fetch and parse an HMRC manual page, returning structured data."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.content, "html.parser")

        # Extract title
        title_tag = soup.find("h1")
        manual_code = extract_manual_code(url)
        title = f"{manual_code} – {title_tag.get_text(strip=True) if title_tag else 'Untitled'}"

        # Extract content blocks (main govspeak containers)
        content_blocks = soup.select("div.gem-c-govspeak, .govuk-govspeak")
        paragraphs = []

        for block in content_blocks:
            for tag in block.find_all(["p", "li", "h2", "h3", "div"]):
                text = tag.get_text(separator=' ', strip=True)
                # Exclude irrelevant FOI notices
                if text and "Freedom of Information Act" not in text:
                    paragraphs.append(clean_text(text))

        full_text = " ".join(paragraphs)
        if len(full_text) < 100:   # Skip very short/empty pages
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
    """Main pipeline: load URLs, fetch each page, save to JSON."""
    if not os.path.exists("output"):
        os.makedirs("output")

    # Read URL list
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
        time.sleep(0.5)  # Be polite: avoid hammering the server

    # Save all results
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} entries to {SAVE_PATH}")

if __name__ == "__main__":
    main()
