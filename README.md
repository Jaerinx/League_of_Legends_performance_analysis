# Predicting a Legend - League of Legends Performance Analysis

By Hieu Nguyen and Palina Volskaya

## Introduction

Our dataset of choice is the team and player data of over 10,000 2022 League of Legends matches, sourced from Oracle's Elixir. As data scietists and as keen video game players, we want to figure out the best strategies for winning. More formally, we want to find the most ideal circumstances for the result to be a win, going through the various stages of data analysis to fully utilize the features in the dataset to make an informed prediction. Our research question was: How do we best predict whether a team will win a League of Legends game, given certain parameters throughout the game?

The original dataset contains 150,588 rows. However, based on the data generation process, each game is reported for each player of each team with two more rows for team statistics. Therefore, the dataset we used aggregated by teams in each game has 25,098 rows.

Relevant columns (features we wanted to explore) include:

- `gameid`: unique integer identifier of each game.
- `teamid`: unique integer identifier of each team.
- `position`: string "team" to indicate team row, or player position in game.
- `result`: 1 or 0 for each row to represent a win or loss, respectively.
- `firstdragon`: 1.0 or 0.0 whether that team has secured the first dragon.
- `league`: string abbreviation for each team's league, which are based on region and serve as preliminary steps for Worlds - the world championship.
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

(Note: columns with 1/0 pairs were converted into booleans during the Data Cleaning process.)

## Data Cleaning and Exploratory Data Analysis

The dataset contained a lot of duplication and unecessary missingness due to rows being in groups of 12 - 5 rows for each player of each teams plus two rows for general team statistics of that game. Therefore, to fully make use of the dataset we divided team and player data into two dataframes based on the `position` column, checking if the row was recorded as a team. For easier manipulation of the dataframe later on, we converted the results and firstdragon columns to booleans. During the machine learning step, we dropped all missing values from the "\_\_at25" columns so we could work with the available data.

Below is the `head` of our cleaned dataframe:

| gameid                | teamid                                  | position | result | firstdragon | goldat25 | xpat25 | league | goldat25 | xpat25 | csat25 | opp_goldat25 | opp_xpat25 | opp_csat25 | golddiffat25 | xpdiffat25 | csdiffat25 | killsat25 | assistsat25 | deathsat25 | opp_killsat25 | opp_assistsat25 | opp_deathsat25 |
| :-------------------- | :-------------------------------------- | :------- | :----- | :---------- | -------: | -----: | :----- | -------: | -----: | -----: | -----------: | ---------: | ---------: | -----------: | ---------: | ---------: | --------: | ----------: | ---------: | ------------: | --------------: | -------------: |
| ESPORTSTMNT01_2690210 | oe:team:733ebb9dbf22a401c0127a0c80193ca | team     | False  | False       |    40224 |  45960 | LCKC   |    40224 |  45960 |    767 |        40136 |      49931 |        864 |           88 |      -3971 |        -97 |         6 |          12 |          7 |             7 |              22 |              6 |
| ESPORTSTMNT01_2690210 | oe:team:7c64febcd5ccff13dcd035dc6867a00 | team     | True   | True        |    40136 |  49931 | LCKC   |    40136 |  49931 |    864 |        40224 |      45960 |        767 |          -88 |       3971 |         97 |         7 |          22 |          6 |             6 |              12 |              7 |
| ESPORTSTMNT01_2690219 | oe:team:731b7a9fd004cdbe2bcb3da795bce47 | team     | False  | False       |    39335 |  49409 | LCKC   |    39335 |  49409 |    895 |        46615 |      57155 |        928 |        -7280 |      -7746 |        -33 |         1 |           1 |          8 |             8 |              13 |              1 |
| ESPORTSTMNT01_2690219 | oe:team:e7a7c6bf58eb268ed3f13aac4158aa8 | team     | True   | True        |    46615 |  57155 | LCKC   |    46615 |  57155 |    928 |        39335 |      49409 |        895 |         7280 |       7746 |         33 |         8 |          13 |          1 |             1 |               1 |              8 |
| 8401-8401_game_1      | oe:team:f4c4528c6981e104a11ea7548630c23 | team     | True   | False       |      nan |    nan | LPL    |      nan |    nan |    nan |          nan |        nan |        nan |          nan |        nan |        nan |       nan |         nan |        nan |           nan |             nan |            nan |

### Univariate Analysis Plot

<iframe src="assets/univariate_analysis.html" width="1200" height="600" frameborder="1"></iframe>

### Bivariate Analysis Plot

<iframe src="assets/bivariate_analysis_winsvsgold.html" width="1200" height="600" frameborder="1"></iframe>
<iframe src="assets/bivariate_analysis_wins_with_without_firstdragon.html" width="1200" height="600" frameborder="1"></iframe>

### Interesting Aggregates

## Assessment of Missingness

### NMAR Analysis

After analyzing the dataset aggregated by teams, we found the `pick1` column to be NMAR due to its inability to be predicted from other columns, and the missingness potentially occuring due to a variable not included in the dataset. Unlike many of the columns we analyzed, the missingness of `pick1` did not depend on league, as many leagues didn't report certain variables as a whole. Instead, 31 out of the 55 leagues had missingness for `pick1` specifically. The missingness also did not depend on `teamid`, as 381 out of 593 teams had a missing value in `pick1`. Instead, the column's missingness depends on the value itself - the champion chosen. Missingness in this column can occur due to the champion being a newly added character to the game, resulting in a null value being recorded. Additional data that would make this column MAR would be data about whether the champion is a newly added character at the time of the game.

### Missingness Dependency

Once we realized that metrics 25 minutes into the game were extremely relevant to our research, we wanted to test which columns could influence their missingness. For our test, we compared the impact of the `gamelength` and `firstdragon` columns on the missingness of the `goldat25` column. We performed two permutation tests based on the following two hypothesis pairs:

`gamelength`:

- **Null Hypothesis:** The missingness of `goldat25` is independent of `gamelength`.
- **Alternate Hypothesis:** The missingness of `goldat25` is associated with `gamelength`.

Our results showed a very low p-value, leading us to reject our null hypothesis. Based on the test, it is likely that the missingness of `goldat25` was associated with `gamelength`. This makes sense logically as games that finish early will be more likely to have unrecorded values past their finish time.

`firstdragon`:

- **Null Hypothesis**: The missingness of `goldat25` is independent of `firstdragon`.
- **Alternate Hypothesis**: The missingness of `goldat25` is associated with `firstdragon`.

Our results returned an insignificant p-value, meaning that we fail to reject the null hypothesis and the missingness of `goldat25` is not influenced by the `firstdragon` column.

## Hypothesis Testing

- **Null Hypothesis:** The win rate between teams who secure the first dragon is equal to the win rate of teams that do not secure the first dragon.
- **Alternate Hypothesis:** The win rate between teams who secure the first dragon is different from the win rate of teams that do not secure the first dragon.
- **Test statistic:** Difference in win rate between games with and without securing the first dragon.

Note: testing at significance level of 0.05.

The permutation test resulted in a p-value of 0.0 which is less than 0.05, we reject the null hypothesis. Based on the data, the win rate of teams who secure the first dragon is significantly higher than teams that do not secure the first dragon.

Since the results were so extreme, (p-value near 0), we can use this in our progress of determining the best circumstances for a win. This test was helpful in determining that `firstdragon` is in fact a very useful feature to consider when predicting win rates.

## Framing a Prediction Problem

Our prediction model is a classifier to predict the game result using the data at the 25 minute mark. Since the response variable could have one of two results - winning and losing - our model is a binary classifier. Our response variable of choice aligns with all our previous intentions so far to find the ideal tactic to win the game. To evaluate our model we will use the F1-score, the data reported may vary by league and result in bias, therefore a balance between precision and recall suits our needs.

## Baseline Model

For our initial baseline model, we decided to use a DecisionTree to predict wins and loses from features `goldat25` and `xpat25`, both quantitative variables so no encoding was necessary. We also did not specify a max_depth. While the model could go as deep as it wanted, it only had a precision and recal of 55%, which is only 5% better than guessing randomly. We knew that this didn't meet our expectations, so we wanted to improve our model to get a better result.

## Final Model

In our final model, we switched from a DecisionTree to a RandomForest, and we added the following hyperparameters - max_depth, number of estimators, max_features, and bootstrapping. Throughout the analysis carried out in the steps prior, we realized that the most informative data occurs at the 25 minute mark. Therefore, expanding our baseline model features, our RandomForest contains all the columns with recorded data at 25 minutes, including `goldat25`, `xpat25`, `csat25`, `opp_goldat25`, `opp_xpat25`, `opp_csat25`, `golddiffat25`, `xpdiffat25`, `csdiffat25`, `killsat25`, `assistsat25`, `deathsat25`, `opp_killsat25`, `opp_assistsat25`, and `opp_deathsat25`. All of these columns are quantitative, and we performed quadratic feature engineering by multiplying together each pair of features, which makes the underlying pattern in the data generating process more clear. Considering the data generation process itself, it is logical that these are the best predictors for the result, since information about the teams' performance mid-game is more indicative of their potential than the other stats recorded in the dataset.

As expected this model greatly outperformed the baseline model, reaching an accuracy of [ ]. The optimal hyperparameters ended up being [ ], [ ]. Therefore, using the statistics at 25 minutes into the game we were able to successfully train a model that can almost always predict whether a team is going to win.

## Fairness Analysis

Throughout various steps of analysis, we realized that a lot of the data reported depends on `league`, as different regions may report information differently. Because of this, we wanted to test the fairness of our model by exploring potential biases arising from differences between leagues. Our Group X was the predicted `result`, and our Group Y was the `league`, and we permuted the `league` column because the model seemed a lot more biased for the LCL league in particular.

- **Null Hypothesis**: The `league` has no association with the quality of the prediction of the `result`.
- **Alternate Hypothesis**: The `league` is associated with a better or worse preduction of the `result`.
- **Test Statistic**: F1-score of the model.
- **Significance Level**: 0.05

Results:

- Our p-value of [ ] is less than alpha of 0.05, so we reject the null hypothesis. This means that our model is in fact likely biased, and that the quality of the predictions depend on which league the data is reported for. While unfortunate for the fairness of our model, it is realistic for real-world data since the data was collected by thousands of different people in different areas of the world.
