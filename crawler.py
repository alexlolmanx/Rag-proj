import asyncio
import json
from playwright.async_api import async_playwright

async def get_uuids():
    async with async_playwright() as p:
        
        browser = await p.chromium.launch(headless=False) 
        context = await browser.new_context()
        page = await context.new_page()
        
        all_uuids = set()
        
        total_pages = 1019 

        print("🚀 ვიწყებ UUID-ების ამოღებას გვერდების მიხედვით...")

        for page_num in range(1, total_pages + 1):
            
            url = f"https://infohub.rs.ge/ka?page={page_num}&pageSize=10"
            print(f"📄 სკანირება: გვერდი {page_num} -> {url}")
            
            try:
               
                await page.goto(url, wait_until="networkidle", timeout=60000)
                
               
                await page.wait_for_timeout(2000)

                
                links = await page.eval_on_selector_all(
                    "a[href*='/workspace/document/']", 
                    "elements => elements.map(e => e.href)"
                )

                if not links:
                    print(f"   ⚠️ გვერდზე {page_num} ლინკები ვერ მოიძებნა.")
                    continue

                found_this_page = 0
                for href in links:
                    
                    parts = href.split('/')
                    
                    uuid = parts[-1].split('?')[0]
                    
                    if len(uuid) >= 32: 
                        all_uuids.add(uuid)
                        found_this_page += 1
                
                print(f"ნაპოვნია {found_this_page} UUID. (ჯამში უნიკალური: {len(all_uuids)})")

            except Exception as e:
                print(f"შეცდომა {page_num} გვერდზე: {e}")

        
        with open("all_ids.json", "w", encoding="utf-8") as f:
            json.dump(list(all_uuids), f, indent=2)

        print(f"\n მზად არის! 'all_ids.json'-ში შენახულია {len(all_uuids)} ID.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(get_uuids())