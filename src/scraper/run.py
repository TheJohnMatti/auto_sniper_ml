import asyncio
import csv
import os
from datetime import datetime
from playwright.async_api import async_playwright, Page

# Ensure our raw data directory exists
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MarketplaceScraper:
    def __init__(self, platform_name: str, selectors: dict, location: str = ""):
        self.platform_name = platform_name
        self.selectors = selectors
        self.location = location
        
        # Adjust file name to include location if provided
        file_prefix = f"{self.platform_name}_{self.location}" if self.location else self.platform_name
        self.output_file = os.path.join(
            OUTPUT_DIR, 
            f"{file_prefix}_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.scraped_data = []

    async def _scroll_page(self, page: Page, scrolls: int = 5):
        """
        Handles infinite scrolling to trigger lazy-loaded listings.
        """
        print(f"[*] Scrolling page {scrolls} times to load content...")
        for i in range(scrolls):
            # Using mouse wheel is often more reliable for React virtualized lists like FB
            await page.mouse.wheel(0, 4000)
            # Randomize sleep slightly to avoid bot detection
            await asyncio.sleep(3.0)
            print(f"[*] Scroll {i+1}/{scrolls} done.")

    async def _extract_listings(self, page: Page):
        """
        Extracts data using the platform-specific selectors passed during initialization.
        """
        print(f"[*] Extracting {self.platform_name} data from DOM...")
        
        LISTING_CARD_SELECTOR = self.selectors["card"]
        
        # We use a try/except here so if the page isn't fully loaded, it doesn't crash the whole run
        try:
            # Wait a moment for the cards to render
            await page.wait_for_selector(LISTING_CARD_SELECTOR, timeout=5000)
            cards = await page.locator(LISTING_CARD_SELECTOR).all()
        except Exception as e:
            print(f"[!] Could not find any listings with selector '{LISTING_CARD_SELECTOR}': {e}")
            return

        for card in cards:
            try:
                # Split inner text by newlines as requested for Facebook layout
                card_text = await card.inner_text()
                lines = [line.strip() for line in card_text.split('\n') if line.strip()]
                
                price = lines[0] if len(lines) > 0 else "UNKNOWN_PRICE"
                title = lines[1] if len(lines) > 1 else "UNKNOWN_TITLE"
                
                # Extract href from the first nested <a> tag
                url_locator = card.locator("a").first
                url_suffix = await url_locator.get_attribute("href") if await url_locator.count() > 0 else ""
                
                # Description not available until we click links
                description = ""

                # Construct full URL if needed (some sites use relative paths)
                full_url = f"https://www.{self.platform_name}.com{url_suffix}" if url_suffix.startswith('/') else url_suffix

                self.scraped_data.append({
                    "scraped_at": datetime.now().isoformat(),
                    "platform": self.platform_name,
                    "location": self.location,
                    "raw_title": title.replace('\n', ' ').strip(),
                    "raw_price": price.replace('\n', '').strip(),
                    "raw_description": description.replace('\n', ' ').strip(),
                    "url": full_url
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
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.scraped_data)
        print(f"[+] Saved data to {self.output_file}")

    async def run(self, start_url: str):
        """Main execution loop."""
        async with async_playwright() as p:
            # We launch headed (headless=False) so YOU can see what is happening while debugging.
            # Change to headless=True once your selectors are perfect.
            browser = await p.chromium.launch(headless=False)
            
            # Create a context with a standard User-Agent to avoid immediate bot flags
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            
            print(f"[*] Navigating to {start_url}...")
            await page.goto(start_url)
            
            # --- POPUP HANDLING ---
            print("[*] Waiting for potential popups...")
            try:
                # Sensible selector based on the provided DOM tree: 
                # Targets the clickable wrapper 'div' containing the specific 'i' icon class
                close_btn = page.locator("div.x1i10hfl:has(i.x15mokao)").first
                await close_btn.wait_for(timeout=4000)
                await close_btn.click()
                print("[+] Popup closed successfully via selector.")
                await asyncio.sleep(1)
            except Exception:
                print("[-] No popup detected or skipped by timeout. Attempting Escape fallback...")
                # Escape key is highly effective for React modals if the selector fails
                await page.keyboard.press("Escape")
                await asyncio.sleep(1)
            # ----------------------
            
            # Optional: Add a pause here if you need to manually log in to Facebook
            # await page.pause() 
            
            await self._scroll_page(page, scrolls=3)
            await self._extract_listings(page)
            
            self._save_to_csv()
            await browser.close()


async def main():
    # List of Canadian cities to scrape
    cities = [
        "toronto",
        "vancouver",
        "montreal",
        "calgary",
        "edmonton",
        "ottawa",
        "winnipeg",
        "halifax"
    ]
    
    # Define platform-specific configurations
    platforms = {
        "facebook": {
            # Use {city} as a placeholder to be formatted in the loop
            "url_template": "https://www.facebook.com/marketplace/{city}/cars/",
            "selectors": {
                "card": "[data-virtualized='false']"
            }
        }
        # "kijiji": {
        #     "url_template": "https://www.kijiji.ca/b-cars-trucks/{city}/c174l1700272", # Simplified example
        #     "selectors": {
        #         "card": "YOUR_KIJIJI_CARD_SELECTOR",
        #         ...
        #     }
        # }
    }

    # You can comment out platforms here to test them one by one
    targets_to_run = ["facebook"]

    for platform_name in targets_to_run:
        config = platforms[platform_name]
        
        for city in cities:
            print(f"\n==================================================")
            print(f"[*] Starting scrape for: {platform_name.upper()} - {city.upper()}")
            print(f"==================================================")
            
            # Construct the dynamic URL for the current city
            start_url = config["url_template"].format(city=city)
            
            scraper = MarketplaceScraper(
                platform_name=platform_name, 
                selectors=config["selectors"],
                location=city
            )
            await scraper.run(start_url)
            
            # A 5-10 second sleep between locations to avoid triggering aggressive bot detection
            print(f"[*] Finished {city}. Sleeping for 5 seconds before next location...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())