# Incorporating fine-grained events in stock movement prediction

In this repo, we will be replicating the paper Incorporating fine-grained events in stock movement prediction by Chen et al with some modifications.  

Accurate stock movement prediction help investors make decisions regarding porfolio management, reducing unexpected risks. However, financial data are not only
noisy but nonstationary, making it difficult to predict solely based on history trading data. Other factors including news, competitors can also greatly affect 
stock prices. In the past years, researchers have employed stock-related texts to forecast stock movement and have proposed various ways to extract semantic 
information. 

However, existing approaches are limited as they employ either raw texts directly or coarse-grained <S,P,O> (subject, predicate, object) stucture into stock prediction 
tasks. It has been shown that the <S,P,O> outperofms the raw-text approach, indicating that incorporating event structure may be useful for stock movement prediction. 
Given the limitations of <S,P,O>, which only extract the same type of limited information regardless of different event types, the authors of this paper propose to 
learn a better representation for the texts --- fine-grained events. 


