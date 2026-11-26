# Mini Project 1: Predicting News Article Shares

Predicts how many times a news article will be shared, using 35,680 articles from an online news site. Five regression models are compared against a simple baseline.

## Prerequisites and installation

- Python 3.11

```
conda create -n mp1 python=3.11 -y
conda activate mp1
pip install -r requirements.txt
```

## Running the pipeline

```
python main.py
```

This takes a few minutes. It prints a validation table and a final test result for each of two scenarios.

All settings are in `src/config.yaml`: the split sizes, which columns to drop, and the settings each model tries during tuning. Change them there, not in the scripts.

| File | Purpose |
| --- | --- |
| `eda.ipynb` | Data exploration and the reasoning behind each cleaning decision |
| `main.py` | Runs the pipeline end to end |
| `src/data_preparation.py` | Cleaning and preprocessing |
| `src/model_training.py` | Splitting, training, tuning and scoring |
| `src/config.yaml` | Settings |

## Pipeline flow

1. **Load** the CSV.
2. **Clean** using the decisions from the EDA (table below).
3. **Split** into train (80%), validation (10%) and test (10%).
4. **Preprocess** inside each model pipeline: fill remaining blanks with the median, apply log(1 + x) to numbers, scale them, and one-hot encode the categories. These steps are fitted on the training data only.
5. **Train** five models. Ridge, Lasso, the decision tree and the random forest are tuned with 3-fold cross-validation on the training data.
6. **Compare** all models on the validation set.
7. **Final check** of the best model on the test set, which is used once.

Steps 2 to 7 run twice, once per scenario (see below).

## Key findings from the EDA

- **Shares are heavily skewed.** The median is 1,400, but a few viral articles reach up to 843,300. Models are trained on log(1 + shares) so those few do not dominate.
- **No single clue is strong.** The best pre-publication clues are keyword performance (`kw_avg_avg`) and the shares of linked articles (`self_reference_*`), with rank correlations of 0.17 to 0.25.
- **Weekend articles get more shares** (median about 2,000 on Saturday against 1,300 to 1,400 midweek).
- **Blank does not always mean zero.** Articles with a blank video count average 3,445 shares, between the 0-video group (2,822) and the 1+ video group (4,282). Blank means not recorded.
- **`n_comments` is the strongest clue (0.36), but it is only known after publishing.**

## Two scenarios

Reader comments arrive after an article is published, at the same time as shares. A model that uses them cannot score a draft. So the pipeline reports both cases:

- **A. Before publishing:** `n_comments` is excluded. This is the model an editor can use before an article goes out.
- **B. After publishing:** `n_comments` is included. This is useful once an article is live, for example to decide early which articles to promote.

## Feature handling

| Feature | Handling | Reason |
| --- | --- | --- |
| `ID`, `URL` | Dropped | Labels, not clues |
| `kw_min_min` | Dropped | 58% of rows hold a -1 placeholder. No link to shares |
| `n_comments` | Dropped in scenario A, kept in B | Only known after publishing |
| `num_videos` | Blanks filled with 0, plus a new `num_videos_missing` flag | 47% blank. Blank means not recorded |
| `data_channel` | Blanks become a `no_topic` category | 15% blank. They behave like their own group (median 1,900 shares) |
| `num_imgs`, `num_hrefs`, `num_self_hrefs`, `self_reference_*` | Blanks filled with the training median | Small gaps (2 to 4%) |
| `kw_avg_min`, `kw_min_avg` | Negatives set to 0 | Share counts cannot be negative |
| All numeric features | log(1 + x), then standard scaling | Most are skewed counts |
| `weekday`, `data_channel` | One-hot encoded | Categories with no order |
| `shares` (target) | Trained on log(1 + shares), predictions converted back | Heavy skew from viral articles |

## Model choices

| Model | Why it is included |
| --- | --- |
| Median baseline | Always predicts the median. The bar every model must beat |
| Linear regression | The simplest model. Easy to explain |
| Ridge | Linear, but shrinks the weights. Helps when clues overlap, as the keyword columns do |
| Lasso | Linear, and can set weak clues to zero |
| Decision tree | Can learn rules a straight line cannot, such as "weekend and social media" |
| Random forest | Averages 200 trees, which is steadier than one tree |

## Evaluation

Metrics:
- **MAE:** the average miss, in shares. The main metric, because it is easy to read and is not dominated by viral articles.
- **RMSE:** like MAE, but large misses count much more.
- **R2 (log scale):** the share of variation explained, measured on log(1 + shares). 0 means no better than a constant guess.

**Scenario A, before publishing (validation set)**

| Model | MAE | RMSE | R2 (log) |
| --- | --- | --- | --- |
| Random forest | 2,262 | 7,779 | 0.146 |
| Linear | 2,296 | 7,821 | 0.093 |
| Lasso | 2,297 | 7,820 | 0.092 |
| Ridge | 2,297 | 7,821 | 0.093 |
| Decision tree | 2,310 | 7,810 | 0.085 |
| Median baseline | 2,331 | 7,975 | -0.062 |

Test set, random forest: MAE 2,305, RMSE 8,002, R2 (log) 0.154. Baseline MAE: 2,380.

**Scenario B, after publishing (validation set)**

| Model | MAE | RMSE | R2 (log) |
| --- | --- | --- | --- |
| Random forest | 1,802 | 6,764 | 0.420 |
| Decision tree | 1,878 | 6,759 | 0.361 |
| Linear | 2,189 | 7,627 | 0.209 |
| Lasso | 2,190 | 7,627 | 0.209 |
| Ridge | 2,190 | 7,627 | 0.209 |
| Median baseline | 2,331 | 7,975 | -0.062 |

Test set, random forest: MAE 1,778, RMSE 6,740, R2 (log) 0.444. Baseline MAE: 2,380.

**Which model is most suitable?**

The random forest is the best model in both scenarios, and its test results match its validation results, so it is not overfitting.

The more important finding is how small the gain is in scenario A. Before publishing, the best model beats the median baseline by only about 3% on MAE. The article's own statistics say little about whether it will be shared. The three linear models are nearly identical, so regularisation adds nothing here.

Once comments are available (scenario B), the random forest cuts the average miss by about 25% against the baseline. The tree models pull ahead of the linear ones, which suggests the link between comments and shares is not a straight line.

Recommendation: use the random forest. Treat a pre-publication prediction as a rough guide only, and refresh it with early comment counts once the article is live.

## Deployment considerations

- **Use the right scenario.** Scenario B's model must not be used on drafts, because the comment count will always be 0 there.
- **The keyword and self-reference features depend on the site's own history.** They must be computed the same way at prediction time as in this dataset.
- **Retrain regularly.** What gets shared changes over time, and `kw_min_min` showed that the site's recording has already changed once.
- **Viral articles stay unpredictable.** RMSE remains high in every model because a few articles get hundreds of thousands of shares. The model should not be used to forecast viral hits.
- **Speed.** Training takes minutes and predicting is instant, so a daily retrain is practical.
