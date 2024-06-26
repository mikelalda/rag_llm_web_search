import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from web_search import *
from argparse import ArgumentParser
import csv
from threading import Thread
import logging
import datetime

class WebSearcher(Thread):
    def __init__(self, input_file, output_file, max_results: int = 10):
        Thread.__init__(self)
        self.input_file = input_file
        self.output_file = output_file
        self.max_results = max_results

        logging.basicConfig(filename="logs/web_search.log", level=logging.INFO)
        # logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Starting de app at %s_%s", datetime.date.fromisoformat('2019-12-04'), datetime.datetime.now().strftime("%H:%M:%S"))

    def search_duckduckgo(self, search: str, max_results: int, instant_answers: bool = True,
                        regular_search_queries: bool = True, get_website_content: bool = False) -> list[dict]:
        query = search.strip("\"'")
        with DDGS() as ddgs:
            if instant_answers:
                answer_list = list(ddgs.answers(query))
            else:
                answer_list = None
            if answer_list:
                answer_dict = answer_list[0]
                answer_dict["search"] = search
                answer_dict["title"] = query
                answer_dict["body"] = answer_dict["text"]
                answer_dict["href"] = answer_dict["url"]
                answer_dict.pop('icon', None)
                answer_dict.pop('topic', None)
                answer_dict.pop('text', None)
                answer_dict.pop('url', None)
                return [answer_dict]
            elif regular_search_queries:
                results = []
                a = ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=max_results)
                for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=max_results):
                    result["search"] = search
                    if get_website_content:
                        result["body"] = self.get_webpage_content(result["href"])
                    results.append(result)
                return results
            else:
                raise ValueError("One of ('instant_answers', 'regular_search_queries') must be True")


    def get_webpage_content(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.content, features="lxml")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        return '\n'.join([s.strip() for s in strings])

    def search(self):
        with open(self.output_file, 'w', newline='') as results:
            fieldnames = ['search','title', 'body', 'href']
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            [writer.writerow({k:v.encode('utf8') for k,v in i.items()}) for i in self.search_duckduckgo(search=u"".join(self.input_file),max_results=self.max_results, get_website_content=True)]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input a text for the scraper", default="Como hacer un diccionario en Python")
    parser.add_argument("-o", "--output", dest="output", help="Output a csv file with results", default="data/web_search_output.csv")
    parser.add_argument("-m", "--max_results", dest="max_results", help="Maximum results for each row", default=5)
    args = parser.parse_args()

    if(args.input == None or args.output == None):
        print("Please provide a query")
    else:
        websearch = WebSearcher(args.input, args.output, args.max_results)
        websearch.start()
        websearch.search()