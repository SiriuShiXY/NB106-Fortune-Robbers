import os
import re
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import sys
from openpyxl.utils.exceptions import IllegalCharacterError

def clean_text(text):
    return ''.join(c for c in text if c.isprintable())

def Crawler(country, target_list):
    headers = {"User-Agent": UserAgent().random} # Set headers with a random user-agent

    # Here we use ShenlongProxy for proxy settings, define the proxy settings
    def fetch_url_content(url, headers):
        proxyHost = "----------"
        proxyPort = 12345
        account = "--------" + country
        password = "--------"
        proxyMeta = f"http://{account}:{password}@{proxyHost}:{proxyPort}"
        proxies = {
            "http": proxyMeta,
            "https": proxyMeta
        }
        
        while True:
            try:
                response = requests.get(url, headers=headers, proxies=proxies, verify=True)
                response.encoding = response.apparent_encoding
                html = response.text
                return html
            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL: {e}")

    def extract_post_publish_times(html_content):
        pattern = r'"post_last_time":"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"'
        matches = re.findall(pattern, html_content)
        return matches

    # Define the crawling parameters
    start_page = 0
    end_page = 0
    date_threshold = datetime.strptime("0000-00-00 00:00:00", "%Y-%m-%d %H:%M:%S")

    user_comments = []
    publish_times = []
    count = 0
    target_length = len(target_list)
    for number in tqdm(range(target_length), desc=f"Crawling targets for {country}"):
        user_comments = []
        publish_times = []
        target = target_list[number]
        terminate = False
        for page in range(start_page, end_page+1):
            if terminate:
                break
            print(f"Target {target} crawling page {page}")

            url = f"https://guba.eastmoney.com/list,{target}_{page}.html"
            html = fetch_url_content(url, headers)

            cur_post_publish_times = extract_post_publish_times(html)
            filtered_post_publish_times = [time for time in cur_post_publish_times if datetime.strptime(time, "%Y-%m-%d %H:%M:%S") > date_threshold]
            if not filtered_post_publish_times:
                terminate = True
                print(f"Target {target} has reached the time threshold")
                break
            publish_times.extend(cur_post_publish_times)
            soup = BeautifulSoup(html, "html.parser")

            comments = soup.find_all("a")
            if len(comments) == 50:
                terminate = True
                print(f"Target {target} has reached the end of the page")
                break

            for comment in comments:
                href = comment.get('href')
                data_posttype = comment.get('data-posttype')
                text = comment.text.strip()

                if href and href != 'javascript:;':
                    if href.startswith("//"):
                        href = "https:" + href
                    elif href.startswith("/"):
                        href = "https://guba.eastmoney.com" + href
                    
                    if data_posttype:
                        user_comments.append([text, href, data_posttype])
                        
                    else:
                        if href and href.startswith("https://i.eastmoney.com"):
                            user_comments[-1].append(text)
                            user_comments[-1].append(href)

                count += 1

                # Rotate user-agent every 35 requests to avoid being blocked
                if count == 35:
                    headers = {"User-Agent": UserAgent().random}
                    count = 0

        # Clean data
        cleaned_user_comments = [[clean_text(item) for item in comment] for comment in user_comments]
        cleaned_publish_times = [clean_text(time) for time in publish_times]

        result = [(comment[0], comment[1], comment[2], comment[3], comment[4], cleaned_publish_times[i]) for i, comment in enumerate(cleaned_user_comments)]
        df = pd.DataFrame(result, columns=['Comment', 'Comment_URL', 'PostType', 'PublishTime', 'Author', 'Author_URL'])
        filename = f"{target}.xlsx"
        try:
            df.to_excel(f"your_target_path/{filename}", index=False)
        except IllegalCharacterError as e:
            print(f"Error writing to Excel: {e}")

if __name__ == "__main__":
    country = sys.argv[1]
    target_list = sys.argv[2:]
    Crawler(country, target_list)
