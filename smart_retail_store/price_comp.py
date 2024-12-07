import requests
from bs4 import BeautifulSoup

def scrape_amazon(product_name):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        product = soup.find("div", {"data-component-type": "s-search-result"})
        price = product.find("span", {"class": "a-price-whole"}).text.strip()
        link = "https://www.amazon.in" + product.find("a", {"class": "a-link-normal"})["href"]

        return {"price": price, "link": link}
    except Exception as e:
        return {"error": f"Amazon scraping failed: {str(e)}"}

def scrape_flipkart(product_name):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        product = soup.find("div", {"class": "_1AtVbE"})
        price = product.find("div", {"class": "_30jeq3"}).text.strip()
        link = "https://www.flipkart.com" + product.find("a")["href"]

        return {"price": price, "link": link}
    except Exception as e:
        return {"error": f"Flipkart scraping failed: {str(e)}"}
