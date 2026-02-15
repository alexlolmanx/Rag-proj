import json
import os
import asyncio
import random
import sys
from playwright.async_api import async_playwright


INPUT_FILE = "all_ids.json"
DATA_FOLDER = "data"

async def scrape_document(context, doc_id):
    url = f"https://infohub.rs.ge/ka/workspace/document/{doc_id}"
    page = await context.new_page()
    
    try:

        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        if "search" in page.url:
            await page.close()
            return False

        
        try:
            await page.wait_for_selector(".document-view, .doc-content, .text-layer", timeout=8000)
        except:
            pass

        await asyncio.sleep(3) 

        
        content = ""
        possible_selectors = [".document-view", ".k-window-content", ".workspace-content"]
        for sel in possible_selectors:
            if await page.locator(sel).count() > 0:
                content = await page.locator(sel).inner_text()
                break
        
        if not content:
            content = await page.locator("body").inner_text()

       
        if "მაჩვენე" in content[-200:] or "დოკუმენტის #: 2400" in content[:300]: 
            await page.close()
            return False

        if len(content) < 500:
            await page.close()
            return False

     
        data = {
            "id": doc_id,
            "document": {
                "content": content,
                "title": f"Document {doc_id}"
            }
        }
        
        filename = f"{DATA_FOLDER}/{doc_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"შენახულია: {doc_id} ({len(content)} სიმბოლო)")
        await page.close()
        return True

    except Exception as e:
        print(f"შეცდომა {doc_id}-ზე: {str(e)[:50]}...")
        await page.close()
        return False

async def main():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    if not os.path.exists(INPUT_FILE):
        print("all_ids.json არ არსებობს!")
        return

    with open(INPUT_FILE, "r") as f:
        all_ids = json.load(f)

    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(all_ids)
    
    current_batch = all_ids[start_idx:end_idx]
    
    print(f"სკრაპერი ჩაირთო დიაპაზონზე: {start_idx} - {end_idx}")
    print(f"დასამუშავებელია {len(current_batch)} დოკუმენტი.")

    async with async_playwright() as p:
        
        browser = await p.chromium.launch(headless=True) 
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        for doc_id in current_batch:
            
            if os.path.exists(f"{DATA_FOLDER}/{doc_id}.json"):
                continue
                
            await scrape_document(context, doc_id)
            
            await asyncio.sleep(random.uniform(1, 3)) 
        
        await browser.close()
    print(f"🏁 დიაპაზონი {start_idx}-{end_idx} დასრულდა!")

if __name__ == "__main__":
    asyncio.run(main())

