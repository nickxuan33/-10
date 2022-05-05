import pandas as pd
import nltk
import numpy as np
import urllib
from fake_useragent import UserAgent
import requests
import re
from urllib.request import Request, urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math
from nltk.corpus import stopwords
stopWords = list(set(stopwords.words('english')))
from bs4 import BeautifulSoup


nltk.download('stopwords')

def get_text(url):
    try:
        req = Request(url , headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req,timeout=5).read()
        soup = BeautifulSoup(webpage, "html.parser")
        texts = soup.findAll(text=True)
        res=u" ".join(t.strip() for t in texts if t.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]'])
        return(res)
    except:
        return False

def google_results(keyword, n_results):
    headers = {"User-Agent": "Mozilla/5.0"}
    cookies = {"CONSENT": "YES+cb.20210720-07-p0.en+FX+410"}
    query = keyword
    query = urllib.parse.quote_plus(query) # Format into URL encoding
    number_result = n_results
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    response = requests.get(google_url, headers=headers, cookies=cookies)
    soup = BeautifulSoup(response.text, "html.parser")
    result = soup.find_all('div', attrs = {'class': 'ZINbbc'})
    results=[re.search('\/url\?q\=(.*)\&sa',str(i.find('a', href = True)['href'])) for i in result if "url" in str(i)]
    links=[i.group(1) for i in results if i != None]
    return (links)

google = google_results('百家樂賺錢', 10)
np.savetxt('google.csv', google, fmt="%s")


def tf_idf_analysis(keyword):
    links=google_results(keyword,12)
    text=[]
    for i in links:
        t=get_text(i)
        if t:
            text.append(t)
            
    v = TfidfVectorizer(min_df=2,analyzer='word',ngram_range=(1,2),stop_words=stopWords)
    x = v.fit_transform(text)
    f = pd.DataFrame(x.toarray(), columns = v.get_feature_names_out())
    d=pd.concat([pd.DataFrame(f.mean(axis=0)),pd.DataFrame(f.max(axis=0))],axis=1)
    
    
    tf=pd.DataFrame((f>0).sum(axis=0))
    d=d.reset_index().merge(tf.reset_index(),on='index',how='left')
    d.columns=['word','average_tfidf','max_tfidf','frequency']
#you can comment the following part if you want the number of URLs that the word occurs. The percentage makes sense
#when we have a lot of URLs to check
    d['frequency']=round((d['frequency']/len(text))*100)
    return(d)

x= tf_idf_analysis('百家樂賺錢')
#remove the numbers and sort by max tfidf and get the top20 words
a = x[x['word'].str.isalpha()].sort_values('max_tfidf',ascending=False).head(10) 

print(a)
np.savetxt('myfile.csv', a, fmt="%s")       