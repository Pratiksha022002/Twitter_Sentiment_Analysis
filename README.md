## Context

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . 

The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

## Content

It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)



```python
import re     
import nltk   
import string 
from dateutil import parser 
import numpy as np 
import pandas as pd 
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt


```
## Data Processing
```python
df = pd.read_csv("data.csv", encoding = 'latin', header=None)
```


- Renaming the Columns


```python
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
```


- Trimming the data


```python
df.drop(['id','query','user_id'],axis = 1,inplace = True)
```

```python
df['sentiment'] = df['sentiment'].replace(4,1)
```


```python
sns.countplot(x="sentiment", data=df,palette=['#432371',"#FAAE7B"])
```
<img width="473" alt="Screenshot 2024-02-04 232133" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/f46896ca-5968-4b67-b0b1-d9713b553c07">

- The number of positive and negative sentiments tweets in the dataset is almost equal, so the data is perfectly balanced, so it will make training a model easier because it helps prevent the model from becoming biased towards one class.

## Temporal Analysis
- The date column of dataset contains some useless information such as year, time zone in context of temporal variation of tweets. Make another column ‘timestamp’ which contains only time of the tweet in the format “HH:MM: SS”.


```python
df['date'].str.slice(4,11)
```
- Convert 'date' column to datetime format using dateutil.parser

```python

df['timestamp'] = df['date'].apply(lambda x: parser.parse(x).strftime('%H:%M:%S'))
```

```python
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='H')).size().reset_index(name='tweet_count')
```

```python
df['sentiment'] = df['sentiment'].astype('category')
df_grouped = df.groupby([pd.Grouper(key='timestamp', freq='H'), 'sentiment']).size().unstack(fill_value=0).reset_index()
sentiment_categories = df['sentiment'].cat.categories
df_grouped['total_tweets'] = df_grouped[sentiment_categories].sum(axis=1)
```

- Temporal analysis of sentiments over time.

```python
plt.figure(figsize=(12, 6))

for sentiment_category in sentiment_categories:
    plt.plot(df_grouped['timestamp'], df_grouped[sentiment_category], label=sentiment_category)

plt.plot(df_grouped['timestamp'], df_grouped['total_tweets'], label='Total Tweets', linestyle='--', color='black')

plt.title('Temporal Analysis of Sentiments Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Tweets')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="482" alt="Screenshot 2024-02-04 232209" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/25aeceb6-ac9a-4cdd-99bb-4adcba22f1a6">

```python
df['length'] = df['text'].apply(len)
```


```python
df['length'].plot(bins=100, kind='hist') 
```
<img width="466" alt="Screenshot 2024-02-04 232227" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/13141c40-4eff-414a-9f22-57b5fe29c03e">


```python
df = df[['sentiment','text','length']]
df.describe()
```


- Data Visualisation

```python
sentences = df['text'].tolist()
sentences_string = " ".join(sentences)
```


```python

plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_string))
```

<img width="431" alt="Screenshot 2024-02-10 224156" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/57a15128-9314-48aa-a505-d6f9ac221558">


- Wordcloud for positive sentiment
```python
sentences = df[df['sentiment'] == 1]['text'].tolist()
sentences_string = " ".join(sentences)

plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_string))
```
<img width="406" alt="Screenshot 2024-02-10 224203" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/b1fa8568-3cd8-4147-98aa-5383bcfcc885">

-  Wordcloud for negative sentiment :
  World cloud can be used to estimate words that have been used most in these tweets and same can be done for tweets having positive sentiments and negative sentiments separately. 
```python
sentences = df[df['sentiment'] == 0]['text'].tolist()
sentences_string = " ".join(sentences)

plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_string))
```

<img width="426" alt="Screenshot 2024-02-10 224210" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/5d1752fb-a7d3-4414-b910-d3463efafde2">


```python
import string
string.punctuation
```

   '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


```python
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
```


## Text Cleaning
- It is important to remove stop words, special characters, and URLs from the data to get more specific correlation of content of the tweets to its sentiments. 
```python
df['Clean_TweetText'] = df['text']
df.head()
```


```python
for a in string.punctuation:
    df['Clean_TweetText'] = df['Clean_TweetText'].str.replace(a, "")
```

- Remove Stopwords

```python
def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text
```

```python
df['Clean_TweetText'] = df['Clean_TweetText'].apply(lambda text : remove_stopwords(text.lower()))
```

- Tokenization and Normalization

```python
df['Clean_TweetText'] = df['Clean_TweetText'].apply(lambda x: x.split())
```

```python
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
df['Clean_TweetText'] = df['Clean_TweetText'].apply(lambda x: [stemmer.stem(i) for i in x])

```

- Reform the tokens

```python
df['Clean_TweetText'] = df['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))
```

- Removing words with less than 3 letters
```python
df['Clean_TweetText'] = df['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
```

- Vectorize the text

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(dtype = 'uint8')
df_countvectorizer = vectorizer.fit_transform(df['text'])
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_countvectorizer, df['sentiment'],test_size = 0.25, random_state=0)
```

**MultinomialNB**

```python
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```

```python
from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```
<img width="425" alt="Screenshot 2024-02-10 224438" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/36bff417-24a7-4e13-b7d6-80294ff0048a">

```python
print(classification_report(y_test, y_predict_test))
```
               precision    recall  f1-score   support
               0       0.76      0.82      0.79    199734
               1       0.80      0.74      0.77    200266
    
        accuracy                           0.78    400000
       macro avg       0.78      0.78      0.78    400000
    weighted avg       0.78      0.78      0.78    400000
    


**Logistic Regression**


```python
from sklearn.linear_model import LogisticRegression
LRmodel = LogisticRegression(C =2, max_iter=1000, n_jobs=1)
LRmodel.fit(X_train, y_train)
```

```python
y_predict_test = LRmodel.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```
<img width="424" alt="Screenshot 2024-02-10 224710" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/157d4042-6405-4520-a790-49a560bb7b75">

```python
print(classification_report(y_test, y_predict_test))
```

                  precision    recall  f1-score   support
    
               0       0.80      0.79      0.79    199734
               1       0.79      0.81      0.80    200266
    
        accuracy                           0.80    400000
       macro avg       0.80      0.80      0.80    400000
    weighted avg       0.80      0.80      0.80    400000
    


**BernoulliNB**


```python
from sklearn.naive_bayes import BernoulliNB
BNBmodel = BernoulliNB(alpha = 2)
BNBmodel.fit(X_train, y_train)
```

```python
y_predict_test = BNBmodel.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```
<img width="404" alt="Screenshot 2024-02-10 224730" src="https://github.com/Pratiksha022002/Twitter_Sentiment_Analysis/assets/99002937/8bea06c0-bd23-451e-a571-83a76cbcfa62">

```python
print(classification_report(y_test, y_predict_test))
```

                  precision    recall  f1-score   support
    
               0       0.77      0.81      0.79    199734
               1       0.80      0.76      0.78    200266
    
        accuracy                           0.78    400000
       macro avg       0.78      0.78      0.78    400000
    weighted avg       0.78      0.78      0.78    400000

## Conclusion:
- Consistent Performance:
Logistic Regression, Naive Bayes (BernoulliNB), and Naive Bayes (MultinomialNB) show similar overall performance with accuracies ranging from 78% to 80%.
- Balanced Predictions:
All models exhibit balanced precision and recall for both positive and negative sentiments.
- Feature Importance:
Logistic Regression provides insights into feature importance, valuable for understanding key contributors to sentiment predictions.
