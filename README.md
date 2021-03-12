# Incorporating fine-grained events in stock movement prediction

In this repo, we will be replicating the paper *Incorporating fine-grained events in stock movement prediction* by Chen et al with some modifications.  

Accurate stock movement prediction help investors make decisions regarding porfolio management, reducing unexpected risks. However, financial data are not only
noisy but nonstationary, making it difficult to predict solely based on history trading data. Other factors including news, competitors can also greatly affect 
stock prices. In the past years, researchers have employed stock-related texts to forecast stock movement and have proposed various ways to extract semantic 
information. 

However, existing approaches are limited as they employ either raw texts directly or coarse-grained <S,P,O> (subject, predicate, object) stucture into stock prediction 
tasks. It has been shown that the <S,P,O> outperforms the raw-text approach, indicating that incorporating event structure may be useful for stock movement prediction. 
Given the limitations of <S,P,O>, which only extract the same type of limited information regardless of different event types, the authors of this paper propose to 
learn a better representation for the texts --- fine-grained events. 


This 'tutorial'/intro is divided into the following parts:
1. Data and Data Processing
2. Methodologies and Model Overview 
3. How to train the model
4. How to evaluate the model




## Data and Data Processing 

### Data

We generate our dataset from news texts and stock trade data based on the process described in the paper with small modifications. Our historical stock data include minute-level trade prices in the past 5 years for 86 companies, with information on opening, closing, lowest, highest prices, total trade volume and volume-weighted average trade price for every minute. 

**INSERT STOCK DATA SNIPPET BELOW**

The news data include 29630 news articles categorized to a specific company from 81 companies that are included in the stock data. Each article includes four fields: “title”, “full text”, “URL” and “publish time”. 

To generate the dataset that combines the stock and texts, we only look at the 81 companies that have both stock data and corresponding news information and couple 
the news text with trade data from the day before news happened. Note that if the previous day has no available trading data, we will move one day back until the trade information becomes available. In addition, we generate the event role label for each word in the news text based on the event extraction process in the next section. To create the binary target variable, we simply compare the stock prices at the time of interest with the stock prices at the time when the news happened, and assign 0 or 1 respectively for price dropping and rising. 

To summarize, the i-th sample `(x_i, y_i, e_i, s_i)` from our dataset contains the news text `x_i` as a sequence of words, the minute-level stock summary `y_i` from the day before news happened, `e_i` the event role lable for each word in the news text `x_i`, and the binary stock movement label `s_i`.

### Data Processing 




## Methodologies and Model Overview

The methods proposed by this paper include two main components: 1) Fine-grained events extraction and 2) (M)SSPM model. 
