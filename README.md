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

We adopt the **FinBERT-BiLSTM-Attention** model architecture for sentiment classification, as proposed in [this paper](https://ieeexplore.ieee.org/abstract/document/9581106). All training and inference scripts are located in the folder named *sentiment_classification*. **FinBERT** model can be found in this [Github repo](https://github.com/valuesimplex/FinBERT). 

To train the model, run: 

```bash
python sentiment_classification/model_train.py
```

To use the model to inference, run: 

```bash
python sentiment_classification/model_inference.py
```

## SGP Factor Generation

We implement a Symbolic Genetic Programming (SGP) pipeline to automatically evolve interpretable alpha factors based on historical lagged features. The system uses DEAP for genetic programming and evaluates individuals via a cross-sectional fitness score that combines **TopR**, **Monotonicity**, and **Rank IC**. All related scripts are located in the `SGP` folder.

To train the genetic programming model and generate the **Top-30** symbolic expressions as alpha factors, run:

```bash
python SGP/train_and_generate_sgp_factors.py
```

Then, to merge the generated SGP_Fxx factors with the original cleaned stock data, run:

```bash
python SGP/merge_augmented_factors.py
```

If desired, you can modify or extend the heuristic operators and fitness metrics by editing the ``utils.py`` file in the same folder.