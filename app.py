import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import time
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Utility functions ---

def syllable_count(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    if word.endswith("e"):
        count = max(1, count - 1)
    if count == 0:
        count = 1
    return count

def flesch_reading_ease(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r'\w+', text)
    syllables = sum(syllable_count(w) for w in words)
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
    return round(score, 2)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/18.18362"
]

# --- Define scraping function ---
def scrape_html(url, max_retries=5, initial_delay=1, max_delay=2):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Referer": "https://www.google.com/"
            }
            response = requests.get(url, headers=headers, timeout=delay)
            if response.status_code == 200:
                return response.text
            else:
                st.warning(f"[Attempt {attempt}] Server responded with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.warning(f"[Attempt {attempt}] Error fetching the url: {e}")
        time.sleep(delay)
        delay = min(delay * 2, max_delay)
    st.error(f"Failed to scrape URL after {max_retries} attempts.")
    return None

# --- Parsing function ---
def parse_html(html):
    if html is None:
        return "", "", 0
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    body_tags = soup.find_all(["p", "article", "main"])
    if not body_tags:
        body_tags = soup.find_all("p")
    text = " ".join(tag.get_text(" ", strip=True) for tag in body_tags)
    word_count = len(text.split())
    return title, text, word_count

# --- Initialize model, scaler, vectorizer (load or retrain here) ---
# For demo, these must be preloaded or loaded from your model files
# Here we define minimal dummy initialization to avoid errors,
# Replace these with your proper training/loading logic.

tfidf_vect = TfidfVectorizer(stop_words="english", max_features=300)
scaler = StandardScaler()
clf = LogisticRegression(max_iter=1000, solver="saga", class_weight="balanced", random_state=42)
urls_list = []  # You must define or load your URL list here.
tfidf_arr = np.array([])  # Load or compute TF-IDF matrix accordingly

# --- Real-time address analyzer function ---
def analyze_url(url):
    html = scrape_html(url)
    if not html:
        return {"error": "Failed to scrape URL"}
    title, text, wc = parse_html(html)
    sc = len(text.split("."))
    fr = flesch_reading_ease(text) if text else 0

    # Feature vector creation
    vec = tfidf_vect.transform([text]).toarray()[0]
    tfidf_mean, tfidf_max, tfidf_sum = vec.mean(), vec.max(), vec.sum()
    features = scaler.transform([[wc, sc, fr, tfidf_mean, tfidf_max, tfidf_sum]])
    pred = clf.predict(features)[0]
    thin = wc < 500

    sims = cosine_similarity([vec], tfidf_arr).flatten() if tfidf_arr.size else np.array([])
    similar_pages = [
        {"url": urls_list[idx], "similarity": float(sims[idx])}
        for idx in np.where(sims > 0.8)[0]
        if urls_list[idx] != url
    ] if sims.size else []

    return {
        "url": url,
        "title": title,
        "word_count": wc,
        "sentence_count": sc,
        "readability": fr,
        "quality_label": pred,
        "is_thin_content": thin,
        "similar_pages": similar_pages,
    }

# --- Streamlit UI ---
st.title("SEO Content Quality & Duplicate Detector")

url_input = st.text_input("Enter URL to analyze:")

if st.button("Analyze URL"):
    if url_input:
        result = analyze_url(url_input)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Content Summary")
            st.write(f"Title: {result['title']}")
            st.write(f"Word Count: {result['word_count']}")
            st.write(f"Sentence Count: {result['sentence_count']}")
            st.write(f"Readability (Flesch Score): {result['readability']}")
            st.write(f"Quality Label: {result['quality_label']}")
            st.write(f"Is Thin Content: {result['is_thin_content']}")
            st.subheader("Similar Pages")
            if result["similar_pages"]:
                st.table(result["similar_pages"])
            else:
                st.write("No similar pages found.")
    else:
        st.warning("Please enter a valid URL.")
