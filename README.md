# NB106-Fortune-Robbers

## Comment Scraper

We implement a robust multi-threaded scraping system to collect user comments from [Eastmoney Guba](https://guba.eastmoney.com/) for a list of target stocks. The crawler uses ShenlongProxy for proxy support and rotates user-agents automatically to reduce the risk of being blocked. Collected comments are saved as individual Excel files, each named after the corresponding stock code. All related scripts are located in the *crawler* folder.

To scrape comments for a single stock, run:

```bash
python crawler/crawler.py
```

To scrape multiple stocks in parallel using multithreading, run:

```bash
python crawler/Crawlers_Starter.py
```


## Sentiment Classification

We adopt the **FinBERT-BiLSTM-Attention** model architecture for sentiment classification, as proposed in [this paper](https://ieeexplore.ieee.org/abstract/document/9581106). All training and inference scripts are located in the folder named *sentiment classification*. **FinBERT** model can be found in this [Github repo](https://github.com/valuesimplex/FinBERT). 

To train the model, run: 

```bash
python sentiment classification/model_train.py
```

To use the model to inference, run: 

```bash
python sentiment classification/model_inference.py
```