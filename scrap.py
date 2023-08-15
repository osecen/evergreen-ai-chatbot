################################################################################
### Step 1
################################################################################

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import qdrant_client
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# import tiktoken
# import openai
# import numpy as np
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
# from ast import literal_eval

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "north.imsaindy.org"
full_url = "https://north.imsaindy.org/"


# domain = "www.imsaindy.org"
# full_url = "https://www.imsaindy.org/"
#
domain = "flex.amazon.com"
full_url = "https://flex.amazon.com/"

# domain = "openai.com"
# full_url = "https://openai.com/"
number1 = 0
# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    global number1
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        headers = {'User-Agent': 'Mozilla/5.0'}
        # response = requests.get(url, headers=headers)
        # with urllib.request.urlopen(url) as response:
        with requests.get(url, headers=headers) as response:
            print("url to open-2: " + url)
            # If the response is not HTML, return an empty list
            if not response.headers['content-type'].startswith("text/html"):
                print("response is not html")
                return []

            # Decode the HTML
            html = response.text
            if number1 < 10:
                # print(response.text)
                number1 = number1 + 1
    except Exception as e:
        print("exception is here")
        print(e)
        return []

    # # # Create the HTML Parser and then Parse the HTML to get hyperlinks
    # # parser = HyperlinkParser()
    # #
    # # parser.feed(html)
    # # Parse the HTML with BeautifulSoup
    # soup = BeautifulSoup(html, 'html.parser')
    #
    # # Find all <a> tags and extract the href attribute
    # hyperlinks = [a['href'] for a in soup.find_all('a', href=True)]

    # Parse the HTML with BeautifulSoup

    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Find all <a> tags and extract the href attribute
    hyperlinks = [a.get('href') for a in soup.find_all('a', href=True)]

    # Find all <link> tags and extract the href attribute
    link_urls = [link.get('href') for link in soup.find_all('link', href=True)]

    # Return a list with both hyperlinks and link_urls
    print(hyperlinks + link_urls)
    driver.quit()
    return hyperlinks + link_urls




################################################################################
### Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None
        print("clean link")
        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            print("link" + link)
            if link.startswith("/apps/events/"):
                continue
            elif link.startswith("/"):
                link = link[1:]
            elif (
                    link.startswith("#")
                    or link.startswith("mailto:")
                    or link.startswith("tel:")

            ):
                continue
            print("result link : " + link)
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


################################################################################
### Step 4
################################################################################

def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            print("inside the parser")
            # Get the text from the URL using BeautifulSoup
            # soup = BeautifulSoup(requests.get(url).text, "html.parser")

            options = Options()
            options.headless = True
            driver = webdriver.Chrome(options=options)

            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            driver.quit()

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)
            print("wrote to directory")

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=".\n",
        chunk_size=1536,
        chunk_overlap=200,
        length_function=len
    )

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=500, chunk_overlap=100
    # )
    # text_splitter = CharacterTextSplitter(chunk_size=1800)

    text = text.replace('. ', '.\n')
    chunks = text_splitter.split_text(text)
    # chunks = text_splitter.split_documents(text)

    for chunk in chunks:
        # print(chunk)
        if(len(chunk) > 1536):
            print("next chunk : " + chunk)

    # last_chunk = None
    # for chunk in text_splitter.split_text(text):
    #     current_chunk = chunk[:100]
    #     if last_chunk:
    #         current_chunk = last_chunk[-20:] + current_chunk
    #     print(current_chunk)
    #     last_chunk = current_chunk

    return chunks




### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
### Step 6
################################################################################
# Create a list to store the text files
def create_list_store():
    texts = []

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):
        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    df.head()


def generate_scraped_text():
    # Create a list to store the text files
    text_string = ""

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):
        # Open the file and read the text
        filename = "text/" + domain + "/" + file
        print("filename " + filename)
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            # texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))
            text_string += text

    # text_string = " ".join(" ".join(t) if isinstance(t, tuple) else t for t in texts)
    return text_string


def create_vector_collection():
    load_dotenv()
    print(os.getenv("QDRANT_HOST"))
    print(os.getenv("QDRANT_COLLECTION_NAME"))

    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE
    )
    client.recreate_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config=vectors_config,
    )


def create_qdrant_doc():
    load_dotenv()
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )
    # add documents to the vector store
    chunks = get_text_chunks(generate_scraped_text())
    print(f"chunks len before: {len(chunks)}")
    chunks = list(dict.fromkeys(chunks))
    print(f"chunks len after: {len(chunks)}")



    # with open("myfile-7.txt", "w") as f:
    #     for chunk in chunks:
    #         f.write(chunk)

    # with open("story.txt") as f:
    #     raw_text = f.read()
    #     chinks = get_text_chunks(raw_text)
    #     vector_store.add_texts(chinks)

    # with open('story-chunks.txt', 'w') as f:
    #     for chunk in chinks:
    #         print("next chunk : " + chunk)
    num_chunks = len(chunks)
    chunk_size = math.ceil(num_chunks / 10)

    split_chunks = []
    for i in range(0, num_chunks, chunk_size):
        # split_chunks.append(chunks[i:i+chunk_size])
        print(i)
        vector_store.add_texts(chunks[i:i+chunk_size])

# crawl(full_url)
# create_vector_collection()
# create_qdrant_doc()
