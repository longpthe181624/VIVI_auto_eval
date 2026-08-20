import os
import re
import html
import json
import time
import requests
from pathlib import Path

BASE_API_URL = "https://omapi.vinfastauto.com/fe/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://om.vinfastauto.com",
    "Referer": "https://om.vinfastauto.com/"
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "om_manuals"


def clean_html(raw_html: str) -> str:
    """Converts HTML markup into clean readable plain text."""
    if not raw_html:
        return ""
    # Replace block tags with newlines
    text = re.sub(r'</?(p|li|h1|h2|h3|h4|h5|h6|div|tr|br|section)[^>]*>', '\n', raw_html, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Clean whitespace and empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def parse_menu_nodes(nodes, depth=1):
    """Recursively extracts title hierarchy and text content from menu nodes."""
    extracted = []
    if not nodes or not isinstance(nodes, list):
        return extracted

    for node in nodes:
        name = node.get("name") or ""
        html_content = node.get("html") or node.get("html_web") or ""
        
        # Heading prefix based on depth
        heading = "#" * min(depth, 4)
        if name.strip():
            extracted.append(f"\n{heading} {name.strip()}\n")

        if html_content.strip():
            cleaned_text = clean_html(html_content)
            if cleaned_text:
                extracted.append(cleaned_text)

        # Recursively process child nodes
        childs = node.get("childs")
        if childs and isinstance(childs, list):
            extracted.extend(parse_menu_nodes(childs, depth + 1))

    return extracted


def fetch_and_save_manual(car_model: str, version: str = "2024", lang: str = "vi", country: str = "vn"):
    """Fetches the full Owner's Manual for a car model from VinFast API with retry logic and saves to data/om_manuals/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Fetching VinFast Owner Manual for: Model={car_model}, Version={version}, Lang={lang}...")
    params = {
        "carModel": car_model,
        "version": version,
        "lang": lang,
        "country": country
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{BASE_API_URL}/menu", headers=HEADERS, params=params, timeout=45)
            if response.status_code != 200:
                print(f"  ⚠️ Attempt {attempt}: API returned status code {response.status_code} for {car_model}")
                time.sleep(2)
                continue

            data = response.json()
            if not data.get("success") or not data.get("data"):
                print(f"  ⚠️ Attempt {attempt}: No manual data found for {car_model}")
                time.sleep(2)
                continue

            nodes = data.get("data", [])
            content_lines = parse_menu_nodes(nodes)
            if not content_lines:
                print(f"  ⚠️ Attempt {attempt}: Empty content extracted for {car_model}")
                time.sleep(2)
                continue

            header_block = [
                f"============================================================",
                f"VINFAST OWNER MANUAL (HƯỚNG DẪN SỬ DỤNG SẢN PHẨM XE VINFAST)",
                f"Car Model: {car_model}",
                f"Version: {version}",
                f"Language: {lang.upper()}",
                f"Source URL: https://om.vinfastauto.com/",
                f"============================================================\n"
            ]

            full_document = "\n".join(header_block + content_lines)
            
            filename = f"VinFast_OM_{car_model}_{version}_{lang}.txt"
            file_path = DATA_DIR / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_document)

            word_count = len(full_document.split())
            print(f"  ✅ Saved: {file_path.name} ({word_count:,} words, {len(full_document):,} chars)")
            return file_path

        except Exception as e:
            print(f"  ⚠️ Attempt {attempt}/{max_retries} Exception fetching {car_model}: {e}")
            time.sleep(3)

    print(f"  ❌ Failed to fetch {car_model} ({lang}) after {max_retries} attempts.")
    return None


def crawl_all_vinfast_manuals():
    """Crawls Owner Manuals for key VinFast electric and ICE vehicle models."""
    models_to_crawl = [
        ("VF8", "2024"),
        ("VF8NP", "2026"),
        ("VF9", "2024"),
        ("VF7", "2024"),
        ("VF6", "2024"),
        ("VF5", "2024"),
        ("VF3", "2024"),
        ("VFe34", "2023"),
        ("Fadil", "2019"),
        ("LuxA2.0", "2019"),
        ("LuxSA2.0", "2019")
    ]

    print("\n🚀 Starting VinFast Owner Manual Crawler...")
    saved_files = []

    for model_code, year in models_to_crawl:
        # Crawl Vietnamese version
        f_vi = fetch_and_save_manual(car_model=model_code, version=year, lang="vi", country="vn")
        if f_vi:
            saved_files.append(f_vi)
        
        time.sleep(1)  # Respectful delay between requests

        # Crawl English version
        f_en = fetch_and_save_manual(car_model=model_code, version=year, lang="en", country="vn")
        if f_en:
            saved_files.append(f_en)

        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"🎉 CRAWL COMPLETE! Total Manual Documents Downloaded: {len(saved_files)}")
    print(f"📁 Destination Folder: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    crawl_all_vinfast_manuals()
