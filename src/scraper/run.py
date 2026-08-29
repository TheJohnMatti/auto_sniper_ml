import asyncio
import csv
import os
import re
import urllib.parse
from datetime import datetime

from playwright.async_api import async_playwright, Page

from src.ml.config import load_config

# Ensure our raw data directory exists
OUTPUT_DIR = "data/raw"

# Newest-first so each scan sees fresh listings before the ~100-item wall.
_SORT_PARAM = "sortBy=creation_time_descend"


def build_start_url(loc: dict) -> tuple[str, str]:
    """A `scraping.locations` entry -> (start_url, location_tag).

    `{slug: windsor}`                         -> /marketplace/windsor/cars/
    `{latitude:, longitude:, radius_km:}`     -> /marketplace/category/vehicles?lat&lng&radius
                                                (the only way to target a point
                                                FB has no named route for, e.g. London ON)
    """
    if loc.get("slug"):
        tag = loc.get("name", loc["slug"])
        return f"https://www.facebook.com/marketplace/{loc['slug']}/cars/?{_SORT_PARAM}", tag
    if loc.get("latitude") is not None and loc.get("longitude") is not None:
        tag = loc.get("name") or f"{loc['latitude']}_{loc['longitude']}"
        q = urllib.parse.urlencode({
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "radius": loc.get("radius_km", 65),
            "exact": "false",
        })
        return f"https://www.facebook.com/marketplace/category/vehicles?{q}&{_SORT_PARAM}", tag
    raise ValueError(f"scraping.locations entry needs `slug` or `latitude`/`longitude`: {loc!r}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matches a Facebook Marketplace price token, e.g. "CA$3,200", "$1,500", "CA$12,999.00", "Free"
PRICE_RE = re.compile(r"^(?:CA)?\$[\d,]+(?:\.\d{2})?$|^Free$", re.IGNORECASE)
# Matches a trailing "City, PROV" location line, e.g. "Calgary, AB"
LOCATION_RE = re.compile(r".+,\s*[A-Z]{2}$")


def parse_card_lines(lines: list[str]) -> dict | None:
    """
    Turn a Facebook Marketplace card's text lines into structured fields.

    Observed card layouts (top to bottom):
        [price, title, "City, PROV"]
        [price, original_price, title, "City, PROV"]        (marked-down listing)
        ["Just listed", price, title, "City, PROV"]         (status badge on top)
        [price, title]                                      (location sometimes absent)

    Strategy: the first line that looks like a price anchors everything. Anything
    above it is a status badge (dropped); an immediately following second price is
    the strike-through original. The naive "lines[1] is the title" assumption
    captured the strike-through price as the title on ~20% of rows.
    """
    if not lines:
        return None

    price_idx = next((i for i, ln in enumerate(lines) if PRICE_RE.match(ln)), None)
    if price_idx is None:
        return None

    price = lines[price_idx]

    idx = price_idx + 1
    original_price = ""
    if idx < len(lines) and PRICE_RE.match(lines[idx]):
        original_price = lines[idx]
        idx += 1

    title = lines[idx] if idx < len(lines) else "UNKNOWN_TITLE"

    listing_location = ""
    if len(lines) > idx + 1 and LOCATION_RE.match(lines[-1]):
        listing_location = lines[-1]

    return {
        "raw_price": price,
        "raw_price_original": original_price,
        "raw_title": title,
        "raw_listing_location": listing_location,
    }


class MarketplaceScraper:
    def __init__(self, platform_name: str, selectors: dict, location: str = ""):
        self.platform_name = platform_name
        self.selectors = selectors
        self.location = location

        # Adjust file name to include location if provided
        file_prefix = f"{self.platform_name}_{self.location}" if self.location else self.platform_name
        self.output_file = os.path.join(
            OUTPUT_DIR,
            f"{file_prefix}_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        self.scraped_data = []

    async def _scroll_page(self, page: Page, scrolls: int = 3):
        """
        Handles infinite scrolling to trigger lazy-loaded listings.
        NOTE: Kept to 3 scrolls maximum. Facebook prompts unauthenticated
        users with a hard login wall after exactly 100 listings.
        3 scrolls * ~24 listings/scroll = ~75-90 listings (Safe Zone).
        """
        print(f"[*] Scrolling page {scrolls} times to load content (Avoiding 100-post login wall)...")
        for i in range(scrolls):
            # Using mouse wheel is often more reliable for React virtualized lists like FB
            await page.mouse.wheel(0, 4000)
            # Randomize sleep slightly to avoid bot detection
            await asyncio.sleep(3.0)
            print(f"[*] Scroll {i + 1}/{scrolls} done.")

    # Ordered by robustness: the item-link selector survives FB's class churn far
    # better than the internal data-virtualized attribute.
    _CARD_SELECTORS = (
        "a[href*='/marketplace/item/']",
        "[data-virtualized='false']",
    )

    async def _extract_listings(self, page: Page):
        """Pull listing cards from the rendered feed, trying each known selector."""
        print(f"[*] Extracting {self.platform_name} data from DOM...")

        selectors = [self.selectors["card"], *self._CARD_SELECTORS]
        cards = []
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=5000)
                cards = await page.locator(sel).all()
                if cards:
                    print(f"[*] Matched {len(cards)} cards with {sel!r}")
                    break
            except Exception:
                continue

        if not cards:
            # Almost always a login wall served to a datacenter IP. Dump what we
            # got so it can be inspected from the run artifacts.
            print("[!] No listing cards found - dumping page for inspection "
                  "(login wall / bot gate / selector rot?).")
            try:
                dump = os.path.join(OUTPUT_DIR, f"_debug_{self.location}.html")
                with open(dump, "w", encoding="utf-8") as fh:
                    fh.write(await page.content())
                await page.screenshot(path=dump.replace(".html", ".png"), full_page=False)
                print(f"[!] Wrote {dump} (+ .png)")
            except Exception as e:
                print(f"[!] dump failed: {e}")
            return

        for card in cards:
            try:
                card_text = await card.inner_text()
                lines = [line.strip() for line in card_text.split("\n") if line.strip()]

                parsed = parse_card_lines(lines)
                if parsed is None:
                    continue

                # The card may itself be the <a href="/marketplace/item/..."> or
                # contain one.
                url_suffix = await card.get_attribute("href") or ""
                if "/marketplace/item/" not in url_suffix:
                    link = card.locator("a[href*='/marketplace/item/'], a").first
                    if await link.count() > 0:
                        url_suffix = await link.get_attribute("href") or ""

                # Construct full URL if needed (some sites use relative paths)
                full_url = (
                    f"https://www.{self.platform_name}.com{url_suffix}"
                    if url_suffix.startswith("/")
                    else url_suffix
                )

                self.scraped_data.append({
                    "scraped_at": datetime.now().isoformat(),
                    "platform": self.platform_name,
                    "location": self.location,
                    "raw_title": parsed["raw_title"].replace("\n", " ").strip(),
                    "raw_price": parsed["raw_price"].replace("\n", "").strip(),
                    "raw_price_original": parsed["raw_price_original"].replace("\n", "").strip(),
                    "raw_listing_location": parsed["raw_listing_location"].replace("\n", " ").strip(),
                    # Description not available until we click into each listing
                    "raw_description": "",
                    "url": full_url,
                })
            except Exception as e:
                print(f"[!] Error extracting card: {e}")
                continue

        print(f"[*] Extracted {len(self.scraped_data)} total listings so far.")

    def _save_to_csv(self):
        """Dumps the in-memory list of dicts to a CSV file."""
        if not self.scraped_data:
            print("[!] No data to save.")
            return

        keys = self.scraped_data[0].keys()
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.scraped_data)
        print(f"[+] Saved data to {self.output_file}")

    async def run(self, start_url: str, headless: bool = True):
        """Main execution loop."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)

            # Create a context with a standard User-Agent to avoid immediate bot flags
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )

            page = await context.new_page()

            print(f"[*] Navigating to {start_url}...")
            await page.goto(start_url, timeout=30000)

            # --- POPUP HANDLING ---
            # Facebook throws an unauthenticated "log in" modal over the feed. The
            # old class-based selector (div.x1i10hfl:has(i.x15mokao)) has rotted, so
            # we try a stable aria-label first and fall back to Escape, which
            # reliably dismisses the React modal.
            print("[*] Dismissing login/consent modal if present...")
            try:
                close_btn = page.locator("[aria-label='Close'], [aria-label='Dismiss']").first
                await close_btn.wait_for(timeout=3000)
                await close_btn.click()
                print("[+] Modal closed via aria-label.")
            except Exception:
                print("[-] No aria-label close button; pressing Escape.")
                await page.keyboard.press("Escape")
            await asyncio.sleep(1)
            # ----------------------

            await self._scroll_page(page, scrolls=3)
            await self._extract_listings(page)

            self._save_to_csv()
            await browser.close()


async def main():
    selectors = {"card": "[data-virtualized='false']"}
    locations = load_config()["scraping"].get("locations") or []
    if not locations:
        raise SystemExit("config.yaml scraping.locations is empty - nothing to scrape.")

    for loc in locations:
        start_url, tag = build_start_url(loc)
        print("\n==================================================")
        print(f"[*] Starting scrape: {tag}  ({start_url})")
        print("==================================================")

        scraper = MarketplaceScraper(
            platform_name="facebook",
            selectors=selectors,
            location=tag,
        )
        await scraper.run(start_url)

        print(f"[*] Finished {tag}. Sleeping 5s before next location...")
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
