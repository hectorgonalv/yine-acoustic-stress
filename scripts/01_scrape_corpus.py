# scripts/01_scrape_corpus.py
import argparse
import os
import json
import requests
import re
from bs4 import BeautifulSoup

# Configuration Constants
URL_START = "https://live.bible.is/bible/"
URL_END = "?audio_type=audio"
LANGUAGE_CODE = "PIBWBT"  # Yine code on bible.is

BIBLE_BOOKS = {
    "MAT": 28, "MRK": 16, "LUK": 24, "JHN": 21, "ACT": 28, "ROM": 16,
    "1CO": 16, "2CO": 13, "GAL": 6, "EPH": 6, "PHP": 4, "COL": 4, "1TH": 5,
    "2TH": 3, "1TI": 6, "2TI": 4, "TIT": 3, "PHM": 1, "HEB": 13, "JAS": 5,
    "1PE": 5, "2PE": 3, "1JN": 5, "2JN": 1, "3JN": 1, "JUD": 1, "REV": 22
}

def clean_filename(title):
    """
    Standardize filenames:
    1. Strip leading/trailing whitespace.
    2. Replace spaces with underscores.
    3. Remove any non-alphanumeric characters (safety).
    """
    # Remove special chars (keep only letters, numbers, spaces, underscores)
    safe_title = re.sub(r'[^\w\s]', '', title)
    # Replace spaces with underscores
    clean_title = re.sub(r'\s+', '_', safe_title.strip())
    return clean_title

def scrape_chapters(output_folder):
    """Main scraping loop."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    print(f"Starting scrape for {len(BIBLE_BOOKS)} books (New Testament)...")

    for book_abbr, num_chapters in BIBLE_BOOKS.items():
        print(f"Processing Book: {book_abbr} ({num_chapters} chapters)...")
        
        for chapter_num in range(1, num_chapters + 1):
            try:
                url = f"{URL_START}{LANGUAGE_CODE}/{book_abbr}/{chapter_num}/{URL_END}"
                # Add a user-agent to look like a browser (helps avoid blocks)
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    print(f"  [X] Failed to fetch {book_abbr} {chapter_num} (Status: {response.status_code})")
                    continue

                soup = BeautifulSoup(response.content, "html.parser")
                script_tag = soup.find("script", id="__NEXT_DATA__")

                if not script_tag:
                    print(f"  [X] Data script tag not found for {book_abbr} {chapter_num}")
                    continue

                json_data = json.loads(script_tag.string)
                
                # Safety check for JSON path
                try:
                    chapter_text_list = json_data['props']['pageProps']['chapterText']
                except KeyError:
                    print(f"  [X] JSON structure changed for {book_abbr} {chapter_num}")
                    continue

                if not chapter_text_list:
                    print(f"  [X] No text found in JSON for {book_abbr} {chapter_num}")
                    continue

                # Generate Filename
                raw_title = f"{chapter_text_list[0]['book_name_alt']} {chapter_text_list[0]['chapter_alt']}"
                filename = f"{clean_filename(raw_title)}_text.txt"
                filepath = os.path.join(output_folder, filename)

                # Write to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    for verse_data in chapter_text_list:
                        f.write(verse_data['verse_text'] + "\n")
                
                print(f"  [V] Saved: {filename}")

            except Exception as e:
                print(f"  [!] Error processing {book_abbr} {chapter_num}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Yine Bible text from live.bible.is")
    parser.add_argument("--output_dir", type=str, default="./data/raw", help="Directory to save .txt files")
    
    args = parser.parse_args()
    scrape_chapters(args.output_dir)