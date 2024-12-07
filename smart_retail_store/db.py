import sqlite3
from datetime import datetime
from  price_comp import scrape_amazon, scrape_flipkart

# Price Comparison Functions
def scrape_amazon(product_name):
    # Same as previously defined
    pass

def scrape_flipkart(product_name):
    # Same as previously defined
    pass

# Database initialization
def init_price_db():
    conn = sqlite3.connect("prices.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS price_history (
        product_name TEXT,
        source TEXT,
        price TEXT,
        date TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_price(product_name, source, price):
    conn = sqlite3.connect("prices.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM price_history 
        WHERE product_name = ? AND source = ? AND price = ? 
    """, (product_name, source, price))
    if cursor.fetchone() is None:  # Only insert if no duplicate exists
        cursor.execute("INSERT INTO price_history (product_name, source, price, date) VALUES (?, ?, ?, ?)",
                       (product_name, source, price, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


# Retrieve historical prices
def get_price_history(product_name):
    conn = sqlite3.connect("prices.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM price_history WHERE product_name=?", (product_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Compare prices and fetch history
def price_comparison(product_name):
    amazon_data = scrape_amazon(product_name)
    flipkart_data = scrape_flipkart(product_name)

    # Add to database only if scraping was successful
    if "error" not in amazon_data:
        add_price(product_name, "Amazon", amazon_data["price"])
    if "error" not in flipkart_data:
        add_price(product_name, "Flipkart", flipkart_data["price"])

    # Retrieve historical price data
    history = get_price_history(product_name)

    return {
        "current_prices": {"Amazon": amazon_data, "Flipkart": flipkart_data},
        "history": history,
    }


init_price_db()
