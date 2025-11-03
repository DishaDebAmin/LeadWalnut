# Required Libraries
import pandas as pd
import numpy as np
import requests
import time
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import textstat
import joblib
import json

# 1. Load Dataset
df = pd.read_csv('data.csv')

# 2. Scrape HTML if not present in dataset
def scrape_html(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        time.sleep(1.5)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        print(f"Scraping error for {url}: {e}")
        return None

if 'html' not in df.columns:
    print("Scraping HTML content for urls...")
    df['html'] = df['url'].apply(scrape_html)
    output_csv = 'extracted_html.csv'
    df.to_csv(output_csv, index=False)
else:
    print("HTML content loaded from dataset.")

# 3. Parse HTML to extract title, body text, word count
def parse_html(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        body_tags = soup.find_all(['p', 'article', 'main'])
        if not body_tags:
            body_tags = soup.find_all('p')
        text_content = ' '.join([tag.get_text(separator=' ', strip=True) for tag in body_tags])
        word_count = len(text_content.split())
        return title, text_content, word_count
    except:
        return "", "", 0

print("Parsing HTML content...")
parsed = df['html'].fillna('').apply(parse_html)
df[['title', 'body_text', 'word_count']] = pd.DataFrame(parsed.tolist(), index=df.index)

# Save parsed content
df.to_csv('parsed_content.csv', index=False)

# 4. Feature Extraction
print("Extracting text features...")
df['sentence_count'] = df['body_text'].apply(lambda x: len(x.split('.')))
df['flesch_reading_ease'] = df['body_text'].apply(lambda x: textstat.flesch_reading_ease(x) if x else 0)

tfidf_vect = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vect.fit_transform(df['body_text'].fillna(''))

feature_names = np.array(tfidf_vect.get_feature_names_out())
def get_top_features(row):
    row_arr = row.toarray().flatten()
    top_indices = row_arr.argsort()[-5:][::-1]
    return '|'.join(feature_names[top_indices])
df['top_keywords'] = [get_top_features(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]

df.to_csv('features_extracted.csv', index=False)

# 5. Duplicate Detection by Cosine Similarity
print("Detecting duplicates...")
sim_matrix = cosine_similarity(tfidf_matrix)
threshold = 0.8

duplicates = []
urls = df['url'].tolist()
for i in range(len(urls)):
    for j in range(i+1, len(urls)):
        if sim_matrix[i,j] > threshold:
            duplicates.append({'url1': urls[i], 'url2': urls[j], 'similarity': sim_matrix[i,j]})

duplicates_df = pd.DataFrame(duplicates)
duplicates_df.to_csv('duplicates.csv', index=False)

# Mark thin content pages
df['is_thin'] = df['word_count'] < 500

# 6. Define Quality Labels and Train Classifier
def quality_label(row):
    wc = row['word_count']
    fr = row['flesch_reading_ease']
    if wc > 1500 and 50 <= fr <= 70:
        return 'High'
    elif wc < 500 or fr < 30:
        return 'Low'
    else:
        return 'Medium'

df['quality_label'] = df.apply(quality_label, axis=1)

X = df[['word_count', 'sentence_count', 'flesch_reading_ease']]
y = df['quality_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, 'quality_classifier.pkl')

# 7. Real-time URL Analyzer Function
def analyze_url(url):
    html = scrape_html(url)
    if not html:
        return {"error": "Failed to scrape URL"}

    title, text, wc = parse_html(html)
    sc = len(text.split('.'))
    fr = textstat.flesch_reading_ease(text) if text else 0
    ql = quality_label({'word_count': wc, 'flesch_reading_ease': fr})

    features = np.array([[wc, sc, fr]])
    pred_label = model.predict(features)[0]
    thin = wc < 500

    vec = tfidf_vect.transform([text])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    similar_pages = []
    for idx in np.where(sims > threshold)[0]:
        if urls[idx] != url:
            similar_pages.append({'url': urls[idx], 'similarity': float(sims[idx])})

    result = {
        "url": url,
        "title": title,
        "word_count": wc,
        "sentence_count": sc,
        "readability_score": fr,
        "quality_label": pred_label,
        "is_thin_content": thin,
        "similar_pages": similar_pages
    }
    return result

# Example Run
print(json.dumps(analyze_url("https://example.com/sample-article"), indent=2))
