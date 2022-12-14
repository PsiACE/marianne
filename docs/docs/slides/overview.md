---
template: reveal.html
---

# marianne

#### the best search engine, just for fun

<sup>this slide powered by [reveal.js](https://revealjs.com/) and [mkdocs](https://www.mkdocs.org/)<sup>

---

# What, Why & How

---

## What

**marianne** is a search engine that started from scratch.

> A search engine is a software system designed to carry out web searches.  
> -- [Wikipedia - Search engine](https://en.wikipedia.org/wiki/Search_engine)

--

As we know, Google, Baidu, Bing, these are all search engines.

![google](https://img.republicworld.com/republic-prod/stories/images/15984355715f4630f3709e4.png)

--

Unlike the sophisticated and large commercial systems mentioned above, **marianne** just works. but is also ...

<h5><b>Features</b></h5>

- Designed for the World Wide Web
- Powered by Machine Learning

---

## Why

It has to be admitted that it was born out of homework.

However, there are still a number of <h5>reasons<h5>.

--

- Commercial systems are often large and complex and we need some simpler examples to show how search engines work, for learning purposes.

![arch](https://lh5.googleusercontent.com/-YpfajGcR7CC54g4axauzO9rI9lAz_WT1IITGAgvG2Iq6HZgstL78HzxS-F3J8h09iohW6H_0IXl5gAE0dS57IHzVROFM2zJ3R0IkBdu7AHxdSin0Ev6bDJVQws5A7K_HGUNxgo)

<!-- In this diagram, five computer instances are included for distributed querying and indexing, and a number of copies are included to provide redundancy, but this is only a relatively external view and there is still a lot of detail in it. -->

--

- Hackers might take a stand against the misuse of information gathering and want to try something interesting. Although there is now duckduckgo, but nothing more fun than making some tangible improvements yourself.

![duckduckgo](https://s.yimg.com/uu/api/res/1.2/j1QRSI1uw7AAf0C7IWCjWg--~B/Zmk9ZmlsbDtoPTQ1MDt3PTY3NTthcHBpZD15dGFjaHlvbg--/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2022-04/66fdfcd0-bcfa-11ec-aa4f-5e788b41e9cb.cf.webp)

---

## How

Similar to other search engines, marianne performs this <h5>workflow</h5>

[![xHANCV.png](https://s1.ax1x.com/2022/11/02/xHANCV.png)](https://imgse.com/i/xHANCV)

--

- Crawl submitted url, to obtain webpage metadata
- Analyse text and assign labels
- Search the database and return results to users

---

# Tech Details

---

## Crawl

Almost identical to other crawlers, but with a few differences

--

For world-wide web

![www](https://upload.wikimedia.org/wikipedia/commons/b/b9/WorldWideWebAroundWikipedia.png)

--

Urls need to be dealt with more effectively

```python
def url_sanitize(url):
    """Sanitize url."""
    if (url.startswith('"') and url.endswith('"')) or (
        url.startswith("'") and url.endswith("'")
    ):
        url = url[1:-1]
    ...
    return url
```

<!-- This is due to the fact that we need to deal with a large number of links provided in different forms and crawl it as much as possible. -->

--

Increased need to focus on quality of content

```python
def classify_text(url, desc):
    """Classify text."""
    desc_class = predict_text(desc)
    if desc_class == "spam":
        print("[!] Website may be spam ->", url)
        return "spam"
    else:
        return "ham"
```

<!-- Ultimately we need to present this content to the user as it is, without over-processing. -->

---

## Analyse

Introducing machine learning for basic labelling

--

A plain idea is that we can try to screen pages that may be full of marketing/unwanted information to avoid users taking the risk of losing their assets.

![](https://www.impactbnd.com/hs-fs/hubfs/shutterstock_785999662.jpg?length=1200&name=shutterstock_785999662.jpg)

--

We introduce a random forest model and build a lightweight model, based on the spam text dataset.

```text
 	label 	text
0 	ham 	Go until jurong point, crazy.. Available only ...
1 	ham 	Ok lar... Joking wif u oni...
2 	spam 	Free entry in 2 a wkly comp to win FA Cup fina...
3 	ham 	U dun say so early hor... U c already then say...
4 	ham 	Nah I don't think he goes to usf, he lives aro...
```

--

Random forest is a supervised machine learning algorithm, it is accurate, simple and flexible.

![](https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg)

<!-- In fact, it can be used for classification and regression tasks, and its nonlinear characteristics make it highly adaptable to various data and situations. -->

--

We treat this as a binary classification problem and train and store models for it.

```python
def init_model():
    ...
    vc = CountVectorizer()
    x_train_counts = vc.fit_transform(data["text"])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train_counts, data["label"])
    joblib.dump(
        model, current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.model"
    )
    joblib.dump(vc, current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.vc")
```

<!-- The approximate accuracy is 96%, which may not be particularly good, but it is sufficient. -->

---

## Search

Interacting with users

--

In fact, there are still many details that could be explained here, such as indexing and pagerank, but let's skip those.

```sql
SELECT * FROM METADATA ORDER BY title
```

<!-- Of course, this is just a joke, the process of filtering the results we put in the python code, but let's get to the human interaction part. -->

--

**Search**

[![xHKm60.png](https://s1.ax1x.com/2022/11/02/xHKm60.png)](https://imgse.com/i/xHKm60)

<!-- Similar to other search engines, a basic search box is provided. -->

--

**Result**

[![xHKcct.png](https://s1.ax1x.com/2022/11/02/xHKcct.png)](https://imgse.com/i/xHKcct)

<!-- We can see the results of the search and get hints about the page content labels and, of course, the number of results and pagination. -->

---

## Stack

**marianne** uses <mark>pdm</mark> to manage 

the development environment and ...

--

for <mark>app</mark>

- flask
- joblib
- pandas
- sklearn
- beautifulsoup4

--

for <mark>doc</mark>

- mkdocs
- mkdocs-material
- reveal.js

--

for <mark>lint and test</mark>

- tox
- pre-commit
  - black, flake8, codespell, isort, mypy
- pytest

--

Perhaps we can consider it to be an engineered and well-structured project.

---

# Future

- Better Rank and Index
- More Meaningful Learning
- Better Performance
- Provide Public Sites
