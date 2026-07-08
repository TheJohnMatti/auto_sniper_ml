import asyncio
import os
import glob
from playwright.async_api import async_playwright, Page

OUTPUT_DIR = "data/raw"

async def check_city_route(context, city: str) -> bool:
    """
    Navigates to the city's marketplace URL and checks if it is valid.
    A route is considered invalid if Facebook redirects away from the city name
    or renders an error page.
    """
    url = f"https://www.facebook.com/marketplace/{city}/cars/"
    page = await context.new_page()
    
    try:
        # Wait until the page loads
        await page.goto(url, timeout=15000)
        
        # Check 1: Did Facebook redirect us? (e.g., dropped the city name from the URL)
        if f"/{city}/" not in page.url.lower():
            return False
            
        # Check 2: Did Facebook render a "Not Found" React component?
        # Sometimes the URL stays the same, but the page is an error state
        error_text_count = await page.locator("text=This page isn't available").count()
        if error_text_count > 0:
            return False
            
        return True
    except Exception as e:
        print(f"[!] Error while verifying {city}: {e}")
        return False
    finally:
        await page.close()


async def main():
    # The same list of cities from the main scraper
    cities = [
        "toronto", "ottawa", "mississauga", "hamilton", "london", 
        "kitchener", "waterloo", "windsor", "sudbury", "oshawa", 
        "barrie", "stcatharines", "guelph", "kingston", "thunderbay", 
        "brantford", "peterborough", "niagarafalls",
        "vancouver", "victoria", "kelowna", "nanaimo", "kamloops", "abbotsford",
        "calgary", "edmonton", "reddeer", "lethbridge", "medicinehat",
        "saskatoon", "regina", "winnipeg",
        "montreal", "quebec", "laval", "gatineau", "sherbrooke", "troisrivieres",
        "halifax", "dartmouth", "sydney", "moncton", "saintjohn", 
        "fredericton", "charlottetown", "stjohns"
    ]
    
    invalid_cities = []
    
    print("==================================================")
    print("[*] Starting Marketplace Route Validation")
    print("==================================================")

    async with async_playwright() as p:
        # Run headless so it doesn't interrupt your workflow
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        for city in cities:
            print(f"[*] Verifying route for: {city}...")
            is_valid = await check_city_route(context, city)
            
            if not is_valid:
                print(f"    [-] INVALID ROUTE. Marking {city} for cleanup.")
                invalid_cities.append(city)
                
                # Search for any CSVs belonging to this invalid city and delete them
                search_pattern = os.path.join(OUTPUT_DIR, f"facebook_{city}_raw_*.csv")
                files_to_delete = glob.glob(search_pattern)
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"    [DELETED] {file_path}")
                    except Exception as e:
                        print(f"    [!] Failed to delete {file_path}: {e}")
            else:
                print(f"    [+] Route is valid.")
                
            # Brief pause to respect rate limits
            await asyncio.sleep(1.5)
            
        await browser.close()
        
    print("\n==================================================")
    print(f"[*] Validation Complete. Found {len(invalid_cities)} invalid cities.")
    print("==================================================")
    
    if invalid_cities:
        print("Consider removing these from your main scraper's city list:")
        print(invalid_cities)


if __name__ == "__main__":
    asyncio.run(main())