import pandas as pd
import numpy as np
import requests
import random
import time
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import textstat
import re

def syllable_count(word):
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            syllables += 1
    if word.endswith("e"):
        syllables -= 1
    if syllables == 0:
        syllables = 1
    return syllables

def flesch_reading_ease(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r'\w+', text)
    syllables = sum(syllable_count(word) for word in words)
    num_sentences = len(sentences) if sentences else 1
    num_words = len(words) if words else 1
    # Flesch Reading Ease Formula
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
    return round(score, 2)


# --- List of URLs ---
urls = [
    "https://www.cm-alliance.com/cybersecurity-blog",
    "https://www.varonis.com/blog/cybersecurity-tips",
    "https://www.cisecurity.org/insights/blog/11-cyber-defense-tips-to-stay-secure-at-work-and-home",
    "https://www.cisa.gov/topics/cybersecurity-best-practices",
    "https://www.qnbtrust.bank/Resources/Learning-Center/Blog/7-cyber-security-tips",
    "https://nordlayer.com/learn/network-security/basics/",
    "https://www.fortinet.com/resources/cyberglossary/what-is-network-security",
    "https://www.cisco.com/site/us/en/learn/topics/security/what-is-network-security.html",
    "https://www.trendmicro.com/en_us/what-is/network-security/network-security-basics.html",
    "https://digitdefence.com/blog/fundamentals-of-network-security-in-computer-networks",
    "https://guardiandigital.com/resources/blog/guide-on-phishing",
    "https://cofense.com/blog",
    "https://www.connectwise.com/blog/phishing-prevention-tips",
    "https://www.phriendlyphishing.com/blog",
    "https://inspiredelearning.com/blog/phishing-protection-checklist/",
    "https://en.wikipedia.org/wiki/SD-WAN",
    "https://www.cisco.com/site/us/en/learn/topics/networking/what-is-sd-wan.html",
    "https://www.fortinet.com/resources/cyberglossary/sd-wan-explained",
    "https://www.hpe.com/us/en/what-is/sd-wan.html",
    "https://remotedesktop.google.com/",
    "https://support.microsoft.com/en-us/windows/how-to-use-remote-desktop-5fe128d5-8fb1-7a23-3b8a-41e636865e8c",
    "https://support.apple.com/guide/remote-desktop/welcome/mac",
    "https://en.wikipedia.org/wiki/Remote_desktop_software",
    "https://www.cloudflare.com/learning/access-management/what-is-ztna/",
    "https://www.fortinet.com/solutions/enterprise-midsize-business/network-access/application-access",
    "https://www.microsoft.com/en-us/security/business/security-101/what-is-zero-trust-network-access-ztna",
    "https://www.zscaler.com/resources/security-terms-glossary/what-is-zero-trust-network-access",
    "https://www.efax.com/",
    "https://sign.dropbox.com/products/dropbox-fax",
    "https://www.fax.plus/",
    "https://comfax.com/reviews/free-fax/",
    "https://nytlicensing.com/latest/trends/content-marketing-best-practices-2022/",
    "https://copyblogger.com/content-marketing/",
    "https://www.twilio.com/en-us/blog/insights/content-marketing-best-practices",
    "https://www.akkio.com/beginners-guide-to-machine-learning",
    "https://medium.com/@amitvsolutions/machine-learning-101-the-complete-beginners-guide-to-machine-learning-686a30cbcf6b",
    "https://realpython.com/tutorials/data-science/",
    "https://www.geeksforgeeks.org/data-science/data-science-with-python-tutorial/",
    "https://jakevdp.github.io/PythonDataScienceHandbook/",
    "https://www.w3schools.com/datascience/",
    "https://towardsdatascience.com/machine-learning-basics-with-examples-part-1-c2d37247ec3d",
    "https://www.analyticsvidhya.com/blog/2021/09/comprehensive-guide-on-machine-learning/",
    "https://aws.amazon.com/what-is/deep-learning/",
    "https://developers.google.com/search/docs/fundamentals/seo-starter-guide",
    "https://digitalmarketinginstitute.com/blog/what-is-seo",
    "https://www.coursera.org/articles/content-strategy",
    "https://www.youtube.com/creators/how-things-work/content-creation-strategy/",
    "https://digitalmarketinginstitute.com/blog/what-are-the-most-effective-digital-marketing-strategies",
    "https://emotive.io/blog/11-essential-digital-marketing-tips",
    "https://www.forbes.com/advisor/business/what-is-digital-marketing/",
    "https://blog.hubspot.com/marketing/what-is-digital-marketing",
    "https://www.investopedia.com/terms/s/seo.asp",
    "https://mailchimp.com/marketing-glossary/content-marketing/",
    "https://sproutsocial.com/insights/social-media-marketing-strategy/",
    "https://www.shopify.com/blog/ecommerce-seo-beginners-guide",
    "https://www.dollardays.com/?srsltid=AfmBOopXjdOu2Kwq6fwYN9FPfB19MorSOf5UyS0EisxFSAzOm8wbl8KF",
    "https://www.woot.com/",
    "https://www.shopmissa.com/?srsltid=AfmBOoqr-F6zzKR-vGPqksAZp1wW4niXlINdN2eAREiMVNtHez_0-gAB",
    "https://martie.com/?srsltid=AfmBOoqTEC0QXQ3xYEvV1VWP6OX6EZboYFZDuiiilhyLfDJNbuXIZ97r",
    "https://www.dealsofamerica.com/",
    "https://www.dealnews.com/",
    "https://viewyourdeal.com/",
    "https://americasstealsanddeals.com/",
    "https://www.pricegrabber.com/",
    "https://www.wikihow.com/Make-Money-Online",
    "https://en.wikipedia.org/wiki/Search_engine_optimization",
    "https://simple.wikipedia.org/wiki/Search_engine_optimization",
    "https://en.wikipedia.org/wiki/Content_marketing",
    "https://simple.wikipedia.org/wiki/Content_marketing",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://simple.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://simple.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Digital_marketing",
    "https://simple.wikipedia.org/wiki/Digital_marketing",
    "https://www.reuters.com/technology/artificial-intelligence/",
    "https://www.cnbc.com/artificial-intelligence/",
    "https://www.bbc.com/news/topics/c404v061z99t",
    "https://www.theguardian.com/technology/artificialintelligenceai",
    "https://apnews.com/hub/artificial-intelligence",
    "https://abcnews.go.com/alerts/technology"
]


# --- Robust Scraper with Retries ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6)...Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:106.0)...Firefox/106.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...Edge/18.18362"
]

def scrape_html(url, max_retries=3, timeout=30):
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Non-200 status code {response.status_code} for {url}")
        except Exception as e:
            print(f"Error {e} for {url}, attempt {attempt + 1}")
        wait_time = random.uniform(2, 5)
        print(f"Waiting {wait_time:.1f} seconds before retry...")
        time.sleep(wait_time)
    print(f"Failed to scrape {url} after {max_retries} attempts")
    return None

# --- Load html for all URLs ---
print("Scraping all URLs...")
data = []
for url in urls:
    html = scrape_html(url)
    data.append({'url': url, 'html_content': html})

df = pd.DataFrame(data)
df.to_csv("scraped_urls.csv", index=False)

# --- Parse HTML ---
def parse_html(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        elems = soup.find_all(['p', 'article', 'main']) or soup.find_all('p')
        text = ' '.join([tag.get_text(" ", strip=True) for tag in elems])
        word_count = len(text.split())
        return title, text, word_count
    except Exception as e:
        return "", "", 0

print("Parsing HTML content...")
parsed = df['html_content'].fillna('').apply(parse_html)
df[['title', 'body_text', 'word_count']] = pd.DataFrame(parsed.tolist(), index=df.index)
df.to_csv('parsed_content.csv', index=False)

# --- Feature Engineering ---
print("Extracting features...")
df['sentence_count'] = df['body_text'].apply(lambda x: len(x.split('.')))
df['flesch_reading_ease'] = df['body_text'].apply(lambda x: textstat.flesch_reading_ease(x) if x else 0)
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=300)
tfidf_matrix = tfidf_vect.fit_transform(df['body_text'].fillna(''))
tfidf_arr = tfidf_matrix.toarray()
df['tfidf_mean'] = tfidf_arr.mean(axis=1)
df['tfidf_max'] = tfidf_arr.max(axis=1)
df['tfidf_sum'] = tfidf_arr.sum(axis=1)
df.to_csv('features_extracted.csv', index=False)

# --- Simple Synthetic Labeling (should be replaced by expert labels for production) ---
def quality_label(row):  # tweak as needed for more realistic assignment
    wc, fr = row['word_count'], row['flesch_reading_ease']
    if wc > 1500 and 50 <= fr <= 70:
        return 'High'
    elif wc < 500 or fr < 30:
        return 'Low'
    else:
        return 'Medium'
df['quality_label'] = df.apply(quality_label, axis=1)

# --- Model: Stratified split, regularization, evaluation ---
feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease', 'tfidf_mean', 'tfidf_max', 'tfidf_sum']
X = df[feature_cols]
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=250, C=0.3)
clf.fit(X_train, y_train)
cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
print("CV mean/std:", np.mean(cv_scores), np.std(cv_scores))
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# --- Duplicate Content Detection ---
sim_matrix = cosine_similarity(tfidf_matrix)
threshold = 0.8
urls_list = df['url'].tolist()
duplicates = [
    {'url1': urls_list[i], 'url2': urls_list[j], 'similarity': sim_matrix[i, j]}
    for i in range(len(urls_list)) for j in range(i + 1, len(urls_list))
    if sim_matrix[i, j] > threshold
]
pd.DataFrame(duplicates).to_csv('duplicates.csv', index=False)

# --- Real-time URL Analysis Example ---
def analyze_url(url):
    html = scrape_html(url)
    if not html:
        return {"error": "Failed to scrape URL"}
    title, text, wc = parse_html(html)
    sc = len(text.split('.'))
    fr = flesch_reading_ease(text) if text else 0
    tfidf_vec = tfidf_vect.transform([text]).toarray()[0]
    tfidf_mean, tfidf_max, tfidf_sum = tfidf_vec.mean(), tfidf_vec.max(), tfidf_vec.sum()
    features = np.array([[wc, sc, fr, tfidf_mean, tfidf_max, tfidf_sum]])
    pred_label = clf.predict(features)[0]
    thin = wc < 500
    sims = cosine_similarity([tfidf_vec], tfidf_arr).flatten()
    similar_pages = [{'url': urls_list[idx], 'similarity': float(sims[idx])} for idx in np.where(sims > threshold)[0] if urls_list[idx] != url]
    return {
        "url": url,
        "title": title,
        "word_count": wc,
        "sentence_count": sc,
        "readability": fr,
        "quality_label": pred_label,
        "is_thin_content": thin,
        "similar_pages": similar_pages
    }

# Example:
# print(analyze_url("https://example.com/test-article"))
