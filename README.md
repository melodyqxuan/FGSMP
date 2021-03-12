# Incorporating fine-grained events in stock movement prediction

In this repo, we replicate the paper *Incorporating fine-grained events in stock movement prediction* by Chen et al with some modifications.  

Accurate stock movement prediction help investors make decisions regarding porfolio management, reducing unexpected risks. However, financial data are not only
noisy but nonstationary, making it difficult to predict solely based on history trading data. Other factors including news, competitors can also greatly affect 
stock prices. In the past years, researchers have employed stock-related texts to forecast stock movement and have proposed various ways to extract semantic 
information. 

However, existing approaches are limited as they employ either raw texts directly or coarse-grained <S,P,O> (subject, predicate, object) stucture into stock prediction 
tasks. It has been shown that <S,P,O>-based methods outperforms the raw-text approach, indicating that incorporating event structure may be useful for stock movement prediction. 
Given the limitations of <S,P,O>, which only extract the same type of limited information regardless of different event types, the authors of this paper propose to 
learn a better representation for the texts, termed as fine-grained events. 

Compared to the <S,P,O>-based event extraction, fine-grained event extraction offers finer granularity in providing structured numerical and contextual information associated with a candidate event. Such information has been proven to be critical for investors to make trading decisions. One example regarding the different representation of the news text is given below. 

<img src = '/src/events_structure.png' width="400">

This report is divided into the following parts:
1. Data and Data Processing
   1. Data 
   2. Data Processing
3. Methodologies and Model Overview 
   1. Event extraction
   2. Models
      1. SSPM
      2. MSSPM
4. Reproducibility
   1. Prerequisites
   2. Building dataset 
   3. How to train the model
   4. How to evaluate the model

## Data and Data Processing 

### Data

We generate our dataset from news texts and stock trade data based on the process described in the paper with small modifications. Our historical stock data include minute-level trade prices in the past 5 years for 86 companies, with historical records on opening, closing, lowest, highest prices, total trade volume and volume-weighted average trade price for every minute. 

The news data include 29630 news articles categorized to a specific company from 81 companies that are included in the stock data. Each article includes four fields: “title”, “full text”, “URL” and “publish time”. 

To generate the dataset that combines the stock and texts, we only look at the intersecting 81 companies that with both stock data and news information. For each news report associated with company `C`, we extract the trade history of `C` happening on the publication date of the report, and couple these two sources into one sample point. To avoid information leakage, we only deploit the trading history segment up until the publication time for model training and inference. This procedure is repeated for all news reports except for those happening before 09:10AM and after 14:50PM, in which the full-day trading history of the previous trading day is employed for dataset construction. We refer to such scenarios as out-of-trading news. For each sample, we construct the binary target label by comparing the close price at the publication time (or the closing price of last trading day for out-of-trading news, resp.) with the opening price for the next trading day. Each sample point is further augmented with an event label `e` for fine-grained movement prediction. We defer the construction of `e` to next section. 

To recapitulate, the i-th sample `(x_i, y_i, e_i, s_i)` from our dataset contains the news report `x_i` as a sequence of words, the minute-level stock summary `y_i` from the day before news happened, `e_i` the event role lable for each word in the news text `x_i`, and the binary stock movement label `s_i`.

### Data Processing 

In order to fit the data to the model, further processing on both texts and trade data is required. Following the data construction step proposed in the paper, we aggregate the past 10 minutes of trading history into a single timestamp, constituting a 60-dimensional (i.e. 6 input features x 10 minutes) input vector. This vector is further aggregated by the change rate compared to the previous timestamp, doubling the feature dimension to 120. We pad the trading history with less than 10 minutes of time steps. We refer to this aggregated trading history as `E_y`. For simplicity, we construct the event labels `e_i` using POS tags, as the detailed implementation of the event extraction process is not available to us. 

## Methodologies and Model Overview

The methods proposed by this paper include two main components: 1) Fine-grained events extraction and 2) (M)SSPM model. 

### Event Extraction 

The authors proposed automatic event extraction processing using a predefined dictionary TFED designed by domain experts for stock-related financial events. The dictionary includes 32 event types and the corresponding trigger words and event roles, which are publicly available. The dictionary also include the POS label of the event roles, and the dependency relation between the event types and the necessary/unnecessary label of event roles. Unfortunately, this piece of auxilliary information cannot be accessed. 

The event extraction can be summarized into 4 steps: 1) Extracting the POS tagging and dependency relation from the news using NLP tools 2) Filter event candidates using the trigger words from TFED 3) Matching the POS tagging and dependency relation to locate event roles 4) Assigning event role labels using BIO tagging.

Since we have no access to the auxilliary information to the TFED, we naively employ the POS tagging as event labels in our experiments and allow the users to choose whether to include the event labels as an input to the model. The specification is provided in the reproducibility section.  

### Models

We primarily implement two models, namely **S**tructured **S**tock **P**rediction **M**odel (SSPM) and **M**ulti-task SSPM, that predicts a binary stock movement label based on historical trading data, news report, and event labels. We briefly introduce the mathematical construction of the models below. 

#### SSPM


<img src = '/src/SSPM.png' width="800">

The input data is a set of `N` tuples `{(x_i, y_i, e_i, s_i)}` for `1 <= i <= N`, where `x_i` is the news report sequence, `y_i` the stock trading history,`e_i` the event labels associated with the news report `x_i`, and `s_i` the binary stock movement label. The construction of `e_i` is analogous to that of POS tags, meaning that the sequence length of `e_i` is identical to that of `x_i`. We omit data indices for the rest of this section for clarity. 

The SSPM model first procure variable-length representations for stock and news respectively via a stack of `Bi-LSTM` layers, followed by `Self-Attn` computed over the hidden representations produced by `Bi-LSTM`.
```
S_x = Self-Attn(Bi-LSTM(Embedding(x)))
S_y = Self-Attn(Bi-LSTM(E_y))
```
where `E_y` is the aggregated trading history aforementioned in the data preprocessing section. The event labels are processed through a separate embedding layer:
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

#### MSSPM

<img src = '/src/MSSPM.png' width="800">

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

## Reproducibility

The full model (with event label CRF and large model specification) is trained on an AWS `p3.8xlarge` instance, totaling over 100 million parameters. We provide code snippet for executing a smaller model for the purpose of this report.

### Prerequisites
```
Python > 3.6
torch > 1.5
allennlp
nltk
tqdm
```
### Building dataset
To build dataset from scratch (i.e. `historical_price/` and `news.json`), simply execute 
```
python data_utils.py --build-train-valid --build-pos
```
in which the `--build-train-valid` flag builds the training and validation datasets with a 7:3 split, and `--build-pos` creates dictionary mappings between POS tagging and indices. We provide these mappings in `pos2index.json` and `index2pos.json` in this repository. Processing the dataset takes ~40min on a 32-core Desktop.

### How to train the model
We provide three modes for model training, which can be specified by the flag `--mode` in `main.py`. If `mode == 0`, we train the SSPM model without event (POS) labels; if `mode == 1`, we train the SSPM model with event labels, as described by the paper; if `mode == 2`, we train the MSSPM model with CRF-augmented loss.

To train with each mode respectively, simply execute
```
python main.py --mode 0 --hidden-dim 128 --sent-encoder-layers 1 --stock-encoder-layers 1
python main.py --mode 1 --hidden-dim 128 --sent-encoder-layers 1 --stock-encoder-layers 1 
python main.py --mode 2 --hidden-dim 128 --sent-encoder-layers 1 --stock-encoder-layers 1 
```
These models are memory intensive as they consume >18GB of GPU memory. We also provide other hyperparameter options including `--num-heads` (number of self-attention heads), `--dropout` (dropout probability for Bi-LSTM layers), and standard optimizer hyperparameters. It is expected that the models will not make meaningful decrease at early epochs as fine-tuning ELMO typically leads to performance degradation at early stages of training.

### How to evaluate the model
For the current model configuration, the best performing model on the validation set will be stored in `checkpoint/ckpt.pth`. For evaluation, simply execute
```
python main.py [your last model configuration] --evaluate
```
For example, for the first model trained in the snippet provided above, we run
```
python main.py --mode 0 --hidden-dim 128 --sent-encoder-layers 1 --stock-encoder-layers 1 --evaluate
```
to perform evluation on the validation set. 


## Validation accuracy(%)

Model Mode | Validation Acc(%)
------------ | -------------
mode 0 | 54.21%
mode 1 | 51.78%
mode 2 | 53.65%
