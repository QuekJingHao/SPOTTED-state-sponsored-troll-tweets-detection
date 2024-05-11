# Project SPOTTED: State-sponsored Troll Tweets Detection

#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro / Objective

We fine-tune a pretrained BERTModel, to predict if a tweet is made by an information operative (state-sponsored troll) or by a verified Twitter account. Our purpose is to increase the efficiency of identifying, and disrupting state-sponsored disinformation campaigns for the defense and intelligence community.

<p align="center">  
  <img src="https://github.com/QuekJingHao/SPOTTED-state-sponsored-troll-tweets-detection/blob/main/3 misc/logo.png" width="350" height="350">
</p>


### Methods Used
* Data Cleaning and Aggregation
* Machine Learning
* Data Visualization
* Predictive Modeling
* Natural Language Processing

### Technologies
* Python
* Pandas, PyTorch, BERTModel
* Jupyter
* Google Colab

## Project Description

Information warfare (_information / influence operations / IO_) is defined as the collection of data, and circulation of propaganda to gain strategic advantage over an adversary. The most notible aspect of such operations is the dissemination of disinformation, to exploit existing societal grievances and divisions. These methods aim to sow discord and distrust, influence the population's beliefs and manipulate public perception, steering the messes toward a direction beneficial to the adversary. A common technique of information operatives is to abuse popular social media platforms such as Twitter, Facebook or Instagram that has a very large user base and reach.

Trolls often create accounts impersonating real people. On these accounts, they can post inflammatory messages, attack commentators or spam official verified accounts with disinformation and provocative rhetoric. A favorite social media platform used by operatives is Twitter. Typically, the moderation and flagging of these accounts require human intervention - having to seeve through these tweets manually, or analysing the account's metadata to identify if the are fake or not. This is a labour intensive work, prone to human error, fatigue and objectivity. Instead, we can exploit advances in natural language processing (NLP), to flag out these tweets and highlight abnormal or suspicous accounts. This will allow moderators and authorities to rapidly remove suspected trolls from their platform.

SPOTTED is a fine-tuned BERTModel. By training the pretrained BERTModel on a large dataset of troll and verified users tweets, it can learn to classify tweets into the two categories. The model then can be used to predict for an unseen tweet, whether is it made by a troll, or by a Twitter user.


## Details of Project

This project is implemented using the standard CRISP-DM framework, namely:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modelling
5. Evaluation

We will skip the last step - Deployment. This README file will condense the details in all of the steps. The detailed implementations are described in greater detail in the individual notebooks in the ```4 notebooks``` folder.

This section will go over each of the steps listed above.

### **Data Understanding**

This and the following sections will summarize the key steps in the ```4 notebooks\1 data collection``` directory.

We aim to produce a dataset of **200 000** tweets, which will consist of tweets made by information operatives as well as verified users. The information operations dataset will be sampled from the **Information Operations Archive**, which is the curated from the Twitter Moderation Research Consortium (TMRC). Ths organization oversees moderation and transperacy on Twitter. Towards making Twitter a safer social media platform, they disclose state-linked trolls engaging in online manipulation and information operations. You can assess the website through:

https://transparency.twitter.com/en/reports/moderation-research.html.

As for the tweets made by verified users, it can be tweets made by accounts such as CNA or Straits Times. We do not need to use Twitter's official API - we can exploit the OSINT Python library Twitter Intelligence Tool (TWINT) to download the tweets, bypassing the need for the official API.

The entire data collection process can be summarizes as follows:

<p align="center">  
  <img src="https://github.com/QuekJingHao/google-data-analytics-capstone-project/blob/main/4 Images/db_overall.png" width="800" height="600">
</p>


### **Data Preparation**

This section will go over the curation of the SPOTTED dataset.

In selecting our data, we aim to achieve the following objectives:
1. Sample from the entire repository between 2018 and 2021
   - The reason is simple: state actors change and modify their MO and tradecraft over time. For example, Russian trolls may target the 2016 elections by spreading disinformation. However in 2021, they may attack the botched American withdrawal from Afghanistan, by amplifying the number of American casualties.
2. Select multiple verified Twitter accounts
   - We again should be as representative as possible to cover a large range of topics.
3. Ensure there is no data leakage of validation data into training dataset
   - We should remove any duplicated tweets if present. We should also randomly shuffle the dataset before we split it into training or validation datasets. In this way, we ensure that the none of the tweets used for validation somehow gets into the training dataset. If this happens, then the performance of the model will be much better than expected.

#### Selection of Information Operations Tweets

I have chose to select the following datasets from the IOA, based on the country's geopolitical significance:

<center>

|                                        |                                                |
|:---------------------------------------|:-----------------------------------------------|
| 1.  IRA Oct 2018                       | 11.  Indonesia Feb 2020                        |
| 2.  Iran Oct 2018                      | 12.  China May 2020                            |
| 3.  Russia Jan 2019                    | 13.  Russia May 2020 $\dagger$                 |
| 4.  Venezuela Jun 2019 $\dagger$       | 14.  Thailand Sept 2020                        |
| 5.  Iran Jan 2019 $\dagger$            | 15.  Iran Feb 2020                             |
| 6.  Iran (Set 1) June 2019             | 16.  Russia GRU Feb 2021                       |
| 7.  Iran (Set 2) June 2019             | 17.  Russia IRA Feb 2014                       |
| 8.  China (Set 1) Aug 2019             | 18.  China Changyu Culture Dec 2021 $\dagger$  |
| 9.  China (Set 2) Aug 2019             | 19.  China Xinjiang Dec 2021 $\dagger$         |
| 10. China (Set 3) Sept 2019 $\dagger$  | 20.  Venezuela Dec 2021 $\dagger$              |

$\dagger$ contains multiple CSVs to be merged

</center>


#### Selection of Verified Users Tweets

In selecting the accounts that make up the clean dataset, we have to pay close attention to the IO dataset. This is because the tweets drawn from the verified users must mirror the context and circumstances the operatives are targeting. An exploratory data analysis on the troll datasets reveals some key aspects of the trolls' MO:

1. The earliest information operations dates back to 2009
2. Majority of the trolls target US government, society and politics
3. Majority disguise their tweets as genuine "news", or amplify current events detrimental to US interests and national security
4. Many disguise as genuine persons using Twitter, albeit parroting extreme and diversive political / social views and amplifying existing grieviances

Hence, we in selecting the verified accounts, we aim to mirror the above mentioned dynamic: we shall collect from many verified news media accounts, and include a fraction of US government accounts. However, we should avoid restricting the clean dataset to just focus on the US or international news. So I have thrown in several accounts related to Singapore. Lastly, some accounts related to entertainment, science and technology is included for good measure.  

Below shows the verified Twitter accounts, categorized into 5 different strata:

<center>

|             |                                      |                             |                                  |                               |                          |
|:-----------:|:-------------------------------------|:----------------------------|:---------------------------------|:------------------------------|:--------------------------|
|**[Strata]**   | **US Politics**                      |**US Military**             | **Singapore Government**        | **Entertainment, Science & Tech** | **International News**      |
|**[Accounts]** | President of the United States       | Department of Defense       | PAP                              | Guns n Roses                  | CNN                       |
|             | Vice President of the United States  | US Cyber Command            | WP                               | Metallica                     | BBC World                 |
|             | The White House                      | Defense Intelligence Agency | Ministry of Defense              | NASA                          | New York Times            |
|             | Hillary Clinton                      | Central Intelligence Agency | Republic of Singapore Air Force  | Google                        | Washington Post           |
|             | Barack Obama                         | US Army                     | Gov Singapore                    | SpaceX                        | The Straits Times         |
|             |                                      | US Air Force                | Ministry of Home Affairs         |                               | Channel News Asia         |
|             |                                      | US Navy                     | Ministry of Education            |                               | TODAY Online              |
|             |                                      | US Marine Corps             | Ministry of Foreign Affairs      |                               | Washington Street Journal |
|             |                                      | Indo-Pacific Command        | Ministry of Health               |                               | Reuters                   |
|             |                                      |                             |                                  |                               | The Economist             |
|             |                                      |                             |                                  |                               | Financial Times           |
|             |                                      |                             |                                  |                               | Bloomberg                 |
|             |                                      |                             |                                  |                               | Forbes                    |
|             |                                      |                             |                                  |                               | CNBC                      |
|             |                                      |                             |                                  |                               | MSNBC                     |
|             |                                      |                             |                                  |                               | CBS News                  |
|             |                                      |                             |                                  |                               | ABC                       |
|             |                                      |                             |                                  |                               | CNN International         |
|             |                                      |                             |                                  |                               | New York Times World      |

</center>

The collection of Tweets is done using the OSINT library - Twitter Intelligence Tool (TWINT) - which bypasses the need for Twitter API.

The number of tweets we will sample from each of these strata is given in the following table:

<center>

| Strata                          | Number of tweets to randomly sample    |
|:--------------------------------|:---------------------------------------|
| US Politics                     | 10 000                                 |
| US Military                     | 10 000                                 |
| Singapore Government            | 10 000                                 |
| Entertainment, Science & Tech   | 10 000                                 |
| International News              | 70 000                                 |

</center>

The stratified random sampling process can be summarized in the following chart:


<p align="center">  
  <img src="https://github.com/QuekJingHao/google-data-analytics-capstone-project/blob/main/4 Images/db_overall.png" width="800" height="600">
</p>

Lastly, we encode the IO and verified tweets as follows:

<center>

|                            |      |  
|:--------------------------:|:----:|
| Information Operative:     | 1    |
| Verified:                  | 0    |

</center>


#### Modelling

This section summarizes the implementations in the directory ```4 notebooks/2 model selection```. 

In natural language processing, one needs to obtain the embeddings of a sentence, to be used a feature vectors in a classifcation model. There are several ways to do so, a simple method is to use ```sentence-transformers``` to directly directly obtain the embeddings.

In this project, we have explored two techniques of doing so:

1. Word2Vec
2. BERTModel 

Specifically for this project, we have used the pretrained model - BertForSequenceClassification. This is owing to its seamless integration with PyTorch, and its much higher accuracy than Word2Vec. This model will be used for Evaulation in the next section.



#### Evaluation

This section summarizes the details in ```main.ipynb``` in the directory ```4 notebooks/3 main```, which covers the details of configuring the Google Colab environment, reading, tokenizing and using SPOTTED to predict the unseen tweets.

One has to specify the BERTModel and Tokenizer:

```python
model_name = 'bert-base-uncased'
tokenizer  = BertTokenizer.from_pretrained(model_name)
```

SPOTTED model is loaded using 

```python
SPOTTED_model   = BertForSequenceClassification.from_pretrained(model_path, num_labels = 2)
SPOTTED_trainer = Trainer(SPOTTED_model)
```

To use the model for prediction, it is as simple as

```python
raw_pred, _, _ = SPOTTED_trainer.predict(X_eval_dataset)

# Assign the predited tweets to the dataframe
df_predict['predicted target'] = np.argmax(raw_pred, axis = 1)
df_predict
```

The evaulation metrics of SPOTTED are as follows:

```python
Confusion matrix:
 [[2430  108]
 [ 143 2319]] 

AUC Score : 0.949681974523394
Recall : 0.9419171405361495
Precision : 0.9555006180469716
F1 Score : 0.9486602577214155
Final Accuracy Score : 0.9498
```




## Featured Deliverables

Here, we showcase the data visualisations in the project.

### 1) Word Frequency Distributions


Troll:

<p align="center">  
  <img src="https://github.com/QuekJingHao/SPOTTED-state-sponsored-troll-tweets-detection/blob/main/3 misc/logo.png" width="350" height="350">
</p>



Verified:




### 2) Wordclouds

Troll:




Verified:



### 3) Kmeans Clustering Plots


Troll:




Verified:


### 4) Topic Wordclouds

Troll:

<p float="left">
  <img src="https://github.com/QuekJingHao/SPOTTED-state-sponsored-troll-tweets-detection/blob/main/3 misc/logo.png" width="350" height="350">
  <img src="https://github.com/QuekJingHao/SPOTTED-state-sponsored-troll-tweets-detection/blob/main/3 misc/logo.png" width="350" height="350">
</p>





Verified:









