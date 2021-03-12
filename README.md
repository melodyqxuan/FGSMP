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


We primarily implement two models, namely **S**tructured **S**tock **P**rediction **M**odel (SSPM) and **M**ulti-task SSPM, that predicts a binary stock movement label based on historical trading data, news report, and event labels. We briefly introduce the mathematical construction of the models below. 

## SSPM

The input data is a set of `N` tuples `{(x_i, y_i, e_i, s_i)}` for `1 <= i <= N`, where `x_i` is the news report sequence, `y_i` the stock trading history,`e_i` the event labels associated with the news report `x_i`, and `s_i` the binary stock movement label. The construction of `e_i` is analogous to that of POS tags, meaning that the sequence length of `e_i` is identical to that of `x_i`. We omit data indices for the rest of this section for clarity.

The SSPM model first procure variable-length representations for stock and news respectively via a stack of `Bi-LSTM` layers, followed by `Self-Attn` computed over the hidden representations produced by `Bi-LSTM`.
```
S_x = Self-Attn(Bi-LSTM(Embedding(x)))
S_y = Self-Attn(Bi-LSTM(y))
```
The event labels are processed through a separate embedding layer:
```
E_e = Embedding(e)
```
The SSPM model then proceed to procure an *aggregated* representation by concatenating the event label representation `E_e` and sentence representation `S_x` at each time step. This aggregated transformation is then passed through one layer of non-linearity.
```
H_x = sigma(W[S_x, E_e, S_x - E_e, S_x * E_e])
```
where `*` is performed element-wise. Now, the authors aggregate the stock representation `S_y` and news representation `H_x` via the `Co-Attn` layer, which produces two contextual representations by attending to stock and news respectively:
```
(C_x, C_y) = Co-Attn(H_x, S_y)
```
Owing to the observation that stock history may adversely affect the predictive performance of news report, and vice versa, the authors propose a gating layer, which perform gated sums between the co-attended output and the representation extracted prior to co-attention for stock history and news report respectively.
```
G_x = Gated-Sum(C_x, H_x)
G_y = Gated-Sum(C_y, S_y)
```
The output pair `(G_x, G_y)` are of the same length to those of the original news and stock history length, respectively. We finally apply a temporal-pooling layer to obtain a fixed-size representation for the final binary classification task:
```
G_x <- MaxPool1D(G_x)
G_y <- MaxPool1D(G_y)
```
The final prediction probabilty is computed by
```
p = sigmoid(W[G_x, G_y])
```
and the model is trained with the `BCE` (binary cross-entropy) loss:
```
L_BCE = BCE(p, s)
```
At test time, our model simply takes a test tuple `(x, y, e)` and predicts the movement probability `s`.

## MSSPM

The MSSPM model is very similar to SSPM in its processing of news sequence and stock history inputs, and differs in its treatment of the event labels.
At training time, the MSSPM first procures the news representation
```
S_x = Self-Attn(Bi-LSTM(Embedding(x)))
```
which serves as input to a `CRF` (conditional random field) module to predict the event label `e` associated with the input pair. This procedure induce a `CRF loss`
```
(hat(e), L_CRF) = CRF(S_x, e)
```
At training time, the predicted event labels `hat(e)` are not used for the downstream classification task, but rather the true input `e` is used for better fidelity. At test time, the model uses the predicted output `hat(e)` instead, assuming no knowledge to the event label for better out-of-sample performance. The event labels are processed identically to that in the SSPM model: they are fed into an embedding layer, followed by the fusion operation with the news representation. 
Instead of performing gradient descent over the binary cross entropy loss, the MSSPM optimizes over the aggregated loss
```
(1 - eta) L_CRF + eta L_BCE
```
where `0 < eta < 1` is a training hyperparameter.
