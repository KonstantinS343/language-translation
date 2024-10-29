from fastapi.templating import Jinja2Templates
from fastapi import Request
import json
from playwright.async_api import async_playwright

templates = Jinja2Templates(directory="templates")

class Report:
    
    @classmethod
    async def create_report(cls, request: Request) -> bytes:
        
        with open('input.json', 'r') as f:
           data = json.load(f)
           
        with open('output.json', 'r') as f:
           new_data = json.load(f) 
           
        with open('translation.json', 'r') as f:
           translation = json.load(f) 
           
        translation = ''.join(translation.values())
           
        html_content = templates.TemplateResponse("report.html", {"request": request, 
                                                                  "data": data, 
                                                                  "new_data": new_data,
                                                                  "translation": translation}).body
           
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html_content.decode("utf-8"))
            pdf_path = "report.pdf"
            await page.pdf(path=pdf_path, format="A4", print_background=True, margin={'top': '30px', 'bottom': '30px',
                                                                                      'left': '30px', 'right': '30px'})
            await browser.close()
        
        return pdf_path