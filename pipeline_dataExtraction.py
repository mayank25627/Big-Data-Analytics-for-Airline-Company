import csv
import requests
from bs4 import BeautifulSoup
from dateutil import parser

# Base URL of the airline reviews page
base_url = "https://www.airlinequality.com/airline-reviews/air-india/"

# Function to convert date format


def convert_date(date_str):
    # Parse the date string
    date_obj = parser.parse(date_str, dayfirst=True)
    # Format the date as YYYY-MM-DD
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

# Function to scrape reviews from a given URL


def scrape_reviews(url):
    reviews_data = []
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    review_blocks = soup.find_all(
        "h3", class_="text_sub_header userStatusWrapper")
    for review_block in review_blocks:
        # Extract date
        date = review_block.find(
            "time", itemprop="datePublished").get_text(strip=True)
        # Convert date format
        formatted_date = convert_date(date)
        # Extract and print feedback/comment
        text = review_block.find_next(
            "div", class_="text_content").get_text(strip=True)
        text = text.replace("✅Trip Verified|", "").replace(
            "Not Verified", "").replace("|", "")
        reviews_data.append({"Date": formatted_date, "Review": text})
    return reviews_data

# Function to scrape reviews from multiple pages


def scrape_multiple_pages(base_url, max_pages):
    all_reviews = []
    for page in range(1, max_pages + 1):
        page_url = base_url + "page/{}/".format(page)
        reviews_on_page = scrape_reviews(page_url)
        all_reviews.extend(reviews_on_page)
    return all_reviews


# Specify the maximum number of pages to scrape
max_pages = 100

print("Getting Data From the Resource................................................................")
# Scrape reviews from multiple pages
all_reviews = scrape_multiple_pages(base_url, max_pages)

# Write scraped data to a CSV file
csv_file = "./Pipeline data/air_india_reviews.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Date", "Review"])
    writer.writeheader()
    writer.writerows(all_reviews)

print("Data scraped and saved to:", csv_file)
