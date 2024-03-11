import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from web_search import *
from argparse import ArgumentParser
import csv

def search_duckduckgo(search: str, max_results: int, instant_answers: bool = True,
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
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate',
                                    timelimit=None, max_results=max_results):
                result["search"] = search
                if get_website_content:
                    result["body"] = get_webpage_content(result["href"])
                results.append(result)
            return results
        else:
            raise ValueError("One of ('instant_answers', 'regular_search_queries') must be True")


def get_webpage_content(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = soup.stripped_strings
    return '\n'.join([s.strip() for s in strings])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input a csv file for the scraper", default="input.csv")
    args = parser.parse_args()

    if(args.input == None):
        print("Please provide a query")
    with open(args.input, newline='') as csvfile:
        with open('web_search_results.csv', 'w', newline='') as results:
            reader = csv.reader(csvfile)
            fieldnames = ['search','title', 'body', 'href']
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                [writer.writerow({k:v.encode('utf8') for k,v in i.items()}) for i in search_duckduckgo(search=u"  ".join(row),max_results=15, get_website_content=True)]