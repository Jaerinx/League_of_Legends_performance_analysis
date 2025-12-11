# Predicting a Legend - League of Legends Performance Analysis

By Hieu Nguyen and Palina Volskaya

- [Predicting a Legend - League of Legends Performance Analysis](#predicting-a-legend---league-of-legends-performance-analysis)
  - [Introduction](#introduction)
    - [Column introduction](#column-introduction)
  - [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
    - [Univariate Analysis Plot](#univariate-analysis-plot)
    - [Bivariate Analysis Plot](#bivariate-analysis-plot)
    - [Interesting Aggregates](#interesting-aggregates)
  - [Assessment of Missingness](#assessment-of-missingness)
    - [NMAR Analysis](#nmar-analysis)
    - [Missingness Dependency](#missingness-dependency)
  - [Hypothesis Testing](#hypothesis-testing)
  - [Framing a Prediction Problem](#framing-a-prediction-problem)
  - [Baseline Model](#baseline-model)
  - [Final Model](#final-model)
    - [Fairness Analysis](#fairness-analysis)

## Introduction

Our dataset of choice is the team and player data of over 10,000 2022 League of Legends matches, sourced from Oracle's Elixir. As data scietists and as keen video game players, we want to figure out the best strategies for winning. More formally, we want to find the most ideal circumstances for the result to be a win, going through the various stages of data analysis to fully utilize the features in the dataset to make an informed prediction. Our research question was: How do we best predict whether a team will win a League of Legends game, given certain parameters throughout the game?

The original dataset contains 150,588 rows. However, based on the data generation process, each game is reported for each player of each team with two more rows for team statistics. Therefore, the dataset we used aggregated by teams in each game has 25,098 rows.

Throughout the progress of this paper, we aim to answer to the best of our abilities the central question: **To What extent are we able to reliably predict whether or not a team will win in a given game**

### Column introduction

The dataset (originally containing 164 columns and 150,588 rows) contains comprehensive data gathering of each individual players stats throughout a game: kills, deaths, assists, position, etc. Within this dataset, we will only be exploring a subset of key columns that we believed are central to the question we want to answer:

- `gameid`: unique integer identifier of each game.
- `teamid`: unique integer identifier of each team.
- `teamname`: name of the team (this was used mostly as a more readable form of the teamid)
- `position`: string "team" to indicate team row, or player position in game.
- `result`: 1 or 0 for each row to represent a win or loss, respectively.
- `firstdragon`: 1.0 or 0.0 whether that team has secured the first dragon.
- `league`: string abbreviation for each team's league, which are based on region and serve as preliminary steps for Worlds - the world championship.
- `goldspent`: The total gold spent by teams throughout the entire game
  Our final machine learning model used all of the data available at 25 minutes into the game:
- `goldat25`: total gold aquired by a certain team at 25 minutes into the game.
- `xpat25`: xp aquired by a certain team 25 minutes into the game.
- `csat25`: a team's Creep Score 25 minutes into the game.
- `opp_goldat25`: opponent's gold at 25 minutes into the game.
- `opp_xpat25`: opponent's xp 25 minutes into the game.
- `opp_csat25`: opponent's Creep Score 25 minutes into the game.
- `golddiffat25`: difference between team and opponent's gold 25 minutes into the game.
- `xpdiffat25`: difference in xp between team and opponent 25 minutes into the game.
- `csdiffat25`: difference in Creep Score between team and opponent 25 minutes into the game.
- `killsat25`: team's kills 25 minutes into the game.
- `assistsat25`: team's assists 25 minutes into the game.
- `deathsat25`: team's deaths 25 minutes into the game.
- `opp_killsat25`: opponent's kills 25 minutes into the game.
- `opp_assistsat25`: opponent's assists 25 minutes into the game.
- `opp_deathsat25`: opponent's deaths 25 minutes into the game.

## Data Cleaning and Exploratory Data Analysis

To save processing power and time, we first reduced the dataset to only the relevant columns. Further, it contained a lot of duplication and unnecessary missingness due to rows being in groups of 12 - 5 rows for each player of each teams plus two rows for general team statistics of that game. Therefore, to fully make use of the dataset we divided team and player data into two dataframes based on the `position` column, checking if the row was recorded as a team. As our central question focused more on predicting wins, which is a statistic tied to a team moreso than a player, we only kept the teams table going forward into our exploration. Moreover, during the machine learning step, we dropped all missing values from the "\_\_at25" columns so we could work with the available data.

Below is the `head` of our cleaned dataframe:

| gameid                | teamid                                  | position | result | firstdragon | goldat25 | xpat25 | league | goldat25 | xpat25 | csat25 | opp_goldat25 | opp_xpat25 | opp_csat25 | golddiffat25 | xpdiffat25 | csdiffat25 | killsat25 | assistsat25 | deathsat25 | opp_killsat25 | opp_assistsat25 | opp_deathsat25 |
| :-------------------- | :-------------------------------------- | :------- | :----- | :---------- | -------: | -----: | :----- | -------: | -----: | -----: | -----------: | ---------: | ---------: | -----------: | ---------: | ---------: | --------: | ----------: | ---------: | ------------: | --------------: | -------------: |
| ESPORTSTMNT01_2690210 | oe:team:733ebb9dbf22a401c0127a0c80193ca | team     | False  | False       |    40224 |  45960 | LCKC   |    40224 |  45960 |    767 |        40136 |      49931 |        864 |           88 |      -3971 |        -97 |         6 |          12 |          7 |             7 |              22 |              6 |
| ESPORTSTMNT01_2690210 | oe:team:7c64febcd5ccff13dcd035dc6867a00 | team     | True   | True        |    40136 |  49931 | LCKC   |    40136 |  49931 |    864 |        40224 |      45960 |        767 |          -88 |       3971 |         97 |         7 |          22 |          6 |             6 |              12 |              7 |
| ESPORTSTMNT01_2690219 | oe:team:731b7a9fd004cdbe2bcb3da795bce47 | team     | False  | False       |    39335 |  49409 | LCKC   |    39335 |  49409 |    895 |        46615 |      57155 |        928 |        -7280 |      -7746 |        -33 |         1 |           1 |          8 |             8 |              13 |              1 |
| ESPORTSTMNT01_2690219 | oe:team:e7a7c6bf58eb268ed3f13aac4158aa8 | team     | True   | True        |    46615 |  57155 | LCKC   |    46615 |  57155 |    928 |        39335 |      49409 |        895 |         7280 |       7746 |         33 |         8 |          13 |          1 |             1 |               1 |              8 |
| 8401-8401_game_1      | oe:team:f4c4528c6981e104a11ea7548630c23 | team     | True   | False       |      nan |    nan | LPL    |      nan |    nan |    nan |          nan |        nan |        nan |          nan |        nan |        nan |       nan |         nan |        nan |           nan |             nan |            nan |

Figure 2.1: Head of teams table

### Univariate Analysis Plot

We performed a univariate analysis looking at the frequency of games played for each league.

<figure>
    <iframe src="assets/univariate_analysis.html" width="100%" height="600" frameborder="0"></iframe>
    <figcaption>Figure 2.2: Univariate analysis games played for each league</figcaption>
</figure>

The bar chart shows that the distribution of games across leagues are not uniform: in fact the full range of games between leagues spans `926` games: a significant number larger than most other league game frequencies. This is no-doubt valuable knowledge about the dataset we will be needing for future analysis.

### Bivariate Analysis Plot

To further understand the possible factors that could impact the winrate of a team, we plotted a scatter of the average win-rate of a given team against their average amount of gold spent. To reduce noise in the dataset generated by teams that simply have not played that many matches and therefore become more prone to biases, we filtered the data set for only teams that have played more than 30 matches.

<figure>
    <iframe src="assets/bivariate_analysis_winsvsgold.html" width="100%" height="600" frameborder="0"></iframe>
    <figcaption>Figure 2.3: Bivariate analysis of winrate against gold spent</figcaption>
</figure>

This plot showed us a roughly linear relationship between the win-rate of a team as well as their gold spent. Leading us to want to do further analysis into this.

Further, another major aspect of League of Legends games is whether or not teams are able to attain the first dragon. Attaining the first dragon gives the team a minor but permanent stat boost for the rest of the game, creating a clear advantage for them to win.

<figure>
    <iframe src="assets/bivariate_analysis_wins_with_without_firstdragon.html" width="100%" height="600" frameborder="0"></iframe>
    <figcaption><i>Figure 2.4: Conditional distribution of win rate for first dragons</i></figcaption>
</figure>

And by looking into the conditional distribution of win-rates we can see a (at least qualitatively) significant increase in win-rate for teams that are able to claim the first dragon

### Interesting Aggregates

Here are some interesting aggregates we found:

| first dragon             | # average gold spent | # average kills per death | # average win percentage |
| ------------------------ | -------------------: | ------------------------: | -----------------------: |
| Did NOT get first dragon |             52414.56 |                      1.43 |                    42.16 |
| Got first dragon         |             53313.00 |                      1.94 |                    57.84 |

_Figure 2.5: Interesting aggregates of teams that claim and don't claim the first dragon_

Here, we see that amongst all the statistics: gold spent, kills per death, win percentage, teams that attain the first dragon have a higher average. However, the average kills per death is a signifying column; at this stage we cannot certainly identify that getting the first dragon is really what is increasing these stats, as the teams that simply attain the first dragons are just more likely to be skilled anyway, and thus more likely to win these games.

Looking further into this behaviour, we explored and plotted for each team: their win rate for games where they are able to attain this first dragon and for games in which they are not. We provide below here a sample of our plot:

<figure>
<iframe src="assets\interesting_agg_sample_dragon.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 2.6: Sample of win rate with or without first dragons for each team</i></figcaption>
</figure>

| teamname                             | no_first_dragon_win_pct | first_dragon_win_pct |
| ------------------------------------ | ----------------------- | -------------------- |
| paiN Gaming                          | 0.4230769230769231      | 0.7142857142857143   |
| Spectacled Bears                     | 0.6428571428571429      | 0.64                 |
| Impact Gaming (Latin American Team)  | 0.0                     | 1.0                  |
| Burst The Sky Esports                | 0.30434782608695654     | 0.42857142857142855  |
| UCAM Esports                         | 0.22727272727272727     | 0.6666666666666666   |
| Eclipse Gaming (Latin American Team) | 1.0                     | 0.5                  |
| Sector One                           | 0.5                     | 0.2727272727272727   |
| Riddle Esports                       | 0.46875                 | 0.6774193548387096   |
| God's Plan                           | 0.3                     | 0.5769230769230769   |
| Inside Games                         | 0.2857142857142857      | 0.38461538461538464  |
| Nexus KTRL                           | 0.375                   | 0.4827586206896552   |
| Team Refuse                          | 0.1875                  | 0.32                 |
| INTZ                                 | 0.17391304347826086     | 0.23076923076923078  |
| İstanbul Wildcats                    | 0.5357142857142857      | 0.7714285714285715   |
| Vanir                                | 0.35714285714285715     | 0.36666666666666664  |
| Deliverance Esports Peru             | 0.25                    | 0.14285714285714285  |
| Los Grandes Academy                  | 0.625                   | 0.7142857142857143   |
| IKISEQ                               | 1.0                     | 1.0                  |
| Rare Atom                            | 0.3170731707317073      | 0.5116279069767442   |
| Munster Rugby Gaming                 | 0.5                     | 0.0                  |

_Figure 2.7: Sample of Pivot table deoniting win rate with or without first dragon of each team_

As seen above here, first dragon's impact on the win status of a game is an element of further analysis as this graph seems to be suggesting that for the most part, if you claim the first dragon, you are significantly more likely to win the game.

## Assessment of Missingness

### NMAR Analysis

After analyzing the dataset aggregated by teams, we found the `pick1` column to be NMAR due to its inability to be predicted from other columns, and the missingness potentially occuring due to a variable not included in the dataset. Unlike many of the columns we analyzed, the missingness of `pick1` did not depend on league, as many leagues didn't report certain variables as a whole. Instead, 31 out of the 55 leagues had missingness for `pick1` specifically. The missingness also did not depend on `teamid`, as 381 out of 593 teams had a missing value in `pick1`. Instead, the column's missingness depends on the value itself - the champion chosen. Missingness in this column can occur due to the champion being a newly added character to the game, resulting in a null value being recorded. Additional data that would make this column MAR would be data about whether the champion is a newly added character at the time of the game.

### Missingness Dependency

Once we realized that metrics 25 minutes into the game were extremely relevant to our research, we wanted to test which columns could influence their missingness. For our test, we compared the impact of the `gamelength` and `firstdragon` columns on the missingness of the `goldat25` column. We performed two permutation tests based on the following two hypothesis pairs:

`gamelength`:

- **Null Hypothesis:** The missingness of `goldat25` is independent of `gamelength`.
- **Alternate Hypothesis:** `gamelength` values are significantly shorter where `goldat25` is missing.

Making use of the mean `gamelength` where `goldat25` is missing and permuting the `goldat25` column, our findings showed us a p value of `0`. This lead us to reject our null hypothesis. Based on the test, it is likely that the missingness of `goldat25` was associated with `gamelength`; `gamelength` values seem significantly shorter where `goldat25` is missing. This makes sense logically as games that finish early will be more likely to have unrecorded values past their finish time.

<figure>
<iframe src="assets/missingness_gamelength.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 3.1: Assessment of missingness dependency of goldat25 on gamelength</i></figcaption>
</figure>

`firstdragon`:

- **Null Hypothesis**: `firstdragon's` occur equally as likely regardless of the missingness of `goldat25`.
- **Alternate Hypothesis**: `firstdragon`'s are more likely to occur when `goldat25` is missing.

Our results returned an insignificant p-value: `0.442` , meaning that we fail to reject the null hypothesis and the missingness of `goldat25` is not influenced by the `firstdragon` column.

<figure>
<iframe src="assets/missingness_firstdragon.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 3.1: Assessment of missingness dependency of goldat25 on firstdragon</i></figcaption>
</figure>

## Hypothesis Testing

To further our exploration into the impact of first dragons on the chance of a team winning the game, we decided to test out the hypothesis suggested by _figure 2.4_.

- **Null Hypothesis:** The win rate between teams who secure the first dragon is equal to the win rate of teams that do not secure the first dragon.
- **Alternate Hypothesis:** The win rate between teams who secure the first dragon is significantly higher than the win rate of teams that do not secure the first dragon.
- **Test statistic:** Difference in win rate between games with and without securing the first dragon.

Note: testing at significance level of 0.05.

We ran a permutation test: Permuting over the firstdragon column and calculating the difference in winrate between teams who secure a first dragon and teams who do not. The permutation test resulted in a p-value of 0.0 which is less than 0.05, we reject the null hypothesis. Not only this, basing off the graph, this difference appears to be significantly higher. Based on the data, the win rate of teams who secure the first dragon is significantly higher than teams that do not secure the first dragon.

<figure>
<iframe src="assets/hypothesis_test.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 4.1: Results of first dragon impact on win rate permutation test</i></figcaption>
</figure>

Since the results were so extreme, (p-value near 0), we can use this in our progress of determining the best circumstances for a win. This test was helpful in determining that `firstdragon` is in fact a very useful feature to consider when predicting win rates.

However, in game, first dragons spawn at the 5 minute mark and have no upper bound time limit for when they can be claimed, meaning they can be claimed early game, mid game or even never throughout the game. In fact, within the teams table, the first dragon column contained `2196` missing records, signifying that out of `12549`, 18% of games finish without either team even claiming the first dragon. And thus, therefore as strategists and gamers, we can say that a first dragon would be able to certainly increase our chances of winning. However, as data scientists, whether or not a team claims a first dragon, in terms of the underlying data generating process, cannot be used to predict the results of a game.

## Framing a Prediction Problem

After the initial analysis, in an attempt to look deeper into the underlying data generating process and what impacts results, we wanted to attempt predicting whether a team will be able to win a match based only on their results at the 25 minute mark. Our prediction model is a classifier to predict the game result using the data at the 25 minute mark. Since the response variable could have one of two results - winning and losing - our model is a binary classifier. Our response variable of choice aligns with all our previous intentions so far to find the ideal tactic to win the game. To evaluate our model we will use the F1-score; The data reported, as shown in the univariate analysis plot above vary's largely by league and thus will result in bias, therefore a balance between precision and recall suits our needs and the specifications of this dataset.

## Baseline Model

For our initial baseline model, we decided to use a DecisionTree to predict wins and loses from features `goldat25` and `xpat25`, both quantitative variables so no encoding was necessary. Making use of a standard train-test-split ratio of 20-80 test:train to train our model, we did not specify any specific hyperparameters at this stage such as max_depth. While the model could go as deep as it wanted, it only had a precision of `0.698` and recall of `0.558` with an overall F_1 score of `0.620`: only `.12` better than a "base-case" modal that guesses win for all data points within this data set. We knew that this didn't meet our expectations, so we wanted to improve our model to get a better result.

## Final Model

In our final model, we switched from a DecisionTree to a RandomForest — one that would be much more resilient to overfitting which we will need. We added the following hyperparameters making use of a Grid Search in order to find the optimal parameter values: max_depth (`[None, 5, 10, 20]`), number of estimators ([`[100, 200, 300, 500]`]), max_features ([`'sqrt','log2'`]), and whether or not to bootstrap the data. Throughout the analysis carried out in the steps prior, we realized that the most informative data occurs at the 25 minute mark. Therefore, expanding our baseline model features, our RandomForest contains all the columns with recorded data at 25 minutes, including `goldat25`, `xpat25`, `csat25`, `opp_goldat25`, `opp_xpat25`, `opp_csat25`, `golddiffat25`, `xpdiffat25`, `csdiffat25`, `killsat25`, `assistsat25`, `deathsat25`, `opp_killsat25`, `opp_assistsat25`, and `opp_deathsat25`. All of these columns are quantitative, and we feature engineered crosses between featuresets by multiplying together all pair of features, which makes the underlying pattern in the data generating process more clear and would allow the random forest to better capture the complex relationships between features themselves (such as the relationship between kills and deaths, which interact antagonisticly to us intuitively). Considering the data generation process itself, it is logical that these are the best predictors for the result, since information about the teams' performance mid-game is more indicative of their potential than the other stats recorded in the dataset.

As expected this model greatly outperformed the baseline model, reaching an F-1 score of `0.85`. The optimal hyperparameters ended up being `{'bootstrap': True, 'max_depth': 5, 'max_features': 'log2', 'n_estimators': 100}`. Therefore, using the statistics at 25 minutes into the game we were able to successfully train a model that can predict whether a team is going to win consistently.

### Fairness Analysis

Throughout various steps of analysis, we realized that a lot of the data reported depends on `league`, as different regions may report information differently. Because of this, we wanted to test the fairness of our model by exploring potential biases arising from differences between leagues.Doing a simply Univariate analysis of our model's performance across leagues, There is a clear outlier in the LCL league:

<figure>
<iframe src="assets\Side-by-side_precision_for_each_league.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 7.1: Side by side comparison of F-1 scores separated by league</i></figcaption>
</figure>
Because of this, we decided to further conduct analysis into the significance of this outlier.

Our Group X was the predicted `result`, and our Group Y was the `league`, and we permuted over the `league` column, computing the difference in the F-1 score of the model between games within and not within the LCL league after permutation.

- **Null Hypothesis**: The `league` has no association with the quality of the prediction of the `result`.
- **Alternate Hypothesis**: The model is overperforming for data records where the `league` is LCL.
- **Test Statistic**: difference in F-1 score of the model between games within and not wihin the LCL League
- **Significance Level**: 0.05

<figure>
<iframe src="assets/Distribution of Model F-1.html" width="100%" height="600" frameborder="0"></iframe>
<figcaption><i>Figure 7.3: Results of permutation test of F-1 model performance for LCL and non-LCL games</i></figcaption>
</figure>

Our p-value of `0.001` is less than the alpha of 0.05, so we reject the null hypothesis. This means that the bias seen in our model is likely biased for games within the LCL league. One explanation for this is that the LCL league simply has too little games. Taking a look at _Figure 2.2_, we see that LCL was the league with te least amount of games played, and thus any model fitted over the dataset would too easilly capture the data generating process.
