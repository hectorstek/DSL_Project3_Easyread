import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://globalsymbols.com"
START_URL = "https://globalsymbols.com/symbolsets/arasaac?locale=en"
TOTAL_PAGES = 623

OUTPUT_DIR = "arasaac_images_globalsymbols"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# some filenames occur multiple times
filename_counts = {}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def clean_filename(text):
    """clean the filename"""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)  # remove special chars
    text = re.sub(r"\s+", "_", text)      # spaces -> underscore
    return text


def get_page_url(page):
    if page == 1:
        return START_URL
    return f"{START_URL}&page={page}"


def download_image(url, filename):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

    except Exception as e:
        print(f"Failed: {url} -> {e}")


def process_page(page):
    url = get_page_url(page)
    print(f"Processing page {page}: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page {page}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    cards = soup.find_all("div", class_="col mb-4")

    for card in cards:
        try:
            # image URL is take from style attribute in page source
            picto_div = card.find("div", class_="card-body-picto")
            style = picto_div["style"]

            match = re.search(r"url\((.*?)\)", style)
            if not match:
                continue

            img_url = match.group(1)

            # extract the filename/title from h3 in the page source
            title_tag = card.find("h3")
            title = title_tag.get_text(strip=True)

            clean_name = clean_filename(title)

            # Handle duplicates
            if clean_name in filename_counts:
                filename_counts[clean_name] += 1
                clean_name = f"{clean_name}_{filename_counts[clean_name]}"
            else:
                filename_counts[clean_name] = 1

            filename = os.path.join(OUTPUT_DIR, clean_name + ".png")

            # Download
            download_image(img_url, filename)

            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Skipping item: {e}")

    # avoid getting blocked
    time.sleep(0.5)


def main():
    for page in range(1, TOTAL_PAGES + 1):
        process_page(page)


if __name__ == "__main__":
    main()
