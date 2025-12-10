# Predicting a Legend - League of Legends Performance Analysis
By Hieu Nguyen and Palina Volskaya

## Introduction
Our dataset of choice is the team and player data of over 10,000 2022 League of Legends matches, sourced from Oracle's Elixir. As data scietists and as keen video game players, we want to figure out the best strategies for winning. More formally, we want to find the most ideal circumstances for the result to be a win, going through the various stages of data analysis to fully utilize the features in the dataset to make an informed prediction. Our research question was: How do we best predict whether a team will win a League of Legends game, given certain parameters throughout the game?

The original dataset contains 150,588 rows. However, based on the data generation process, each game is reported for each player of each team with two more rows for team statistics. Therefore, the dataset we used aggregated by teams in each game has 25,098 rows. 

Relevant columns (features we wanted to explore) include:

* `gameid`: unique integer identifier of each game.
* `teamid`: unique integer identifier of each team.
* `position`: string "team" to indicate team row, or player position in game.
* `result`: 1 or 0 for each row to represent a win or loss, respectively.
* `firstdragon`: 1.0 or 0.0 whether that team has secured the first dragon.  
* `league`: string abbreviation for each team's league, which are based on region and serve as preliminary steps for Worlds - the world championship. 
Our final machine learning model used all of the data available at 25 minutes into the game: 
* `goldat25`: total gold aquired by a certain team at 25 minutes into the game. 
* `xpat25`: xp aquired by a certain team 25 minutes into the game. 
* `csat25`: a team's Creep Score 25 minutes into the game. 
* `opp_goldat25`: opponent's gold at 25 minutes into the game. 
* `opp_xpat25`: opponent's xp 25 minutes into the game. 
* `opp_csat25`: opponent's Creep Score 25 minutes into the game. 
* `golddiffat25`: difference between team and opponent's gold 25 minutes into the game. 
* `xpdiffat25`: difference in xp between team and opponent 25 minutes into the game. 
* `csdiffat25`: difference in Creep Score between team and opponent 25 minutes into the game. 
* `killsat25`: team's kills 25 minutes into the game. 
* `assistsat25`: team's assists 25 minutes into the game. 
* `deathsat25`: team's deaths 25 minutes into the game. 
* `opp_killsat25`: opponent's kills 25 minutes into the game. 
* `opp_assistsat25`: opponent's assists 25 minutes into the game. 
* `opp_deathsat25`: opponent's deaths 25 minutes into the game. 

(Note: columns with 1/0 pairs were converted into booleans during the Data Cleaning process.)

## Data Cleaning and Exploratory Data Analysis
The dataset contained a lot of duplication and unecessary missingness due to rows being in groups of 12 - 5 rows for each player of each teams plus two rows for general team statistics of that game. Therefore, to fully make use of the dataset we divided team and player data into two dataframes based on the `position` column, checking if the row was recorded as a team. For easier manipulation of the dataframe later on, we converted the results and firstdragon columns to booleans. During the machine learning step, we dropped all missing values from the "__at25" columns so we could work with the available data. 

Below is the `head` of our cleaned dataframe:
| gameid                | teamid                                  | position   | result   | firstdragon   |   goldat25 |   xpat25 | league   |   goldat25 |   xpat25 |   csat25 |   opp_goldat25 |   opp_xpat25 |   opp_csat25 |   golddiffat25 |   xpdiffat25 |   csdiffat25 |   killsat25 |   assistsat25 |   deathsat25 |   opp_killsat25 |   opp_assistsat25 |   opp_deathsat25 |
|:----------------------|:----------------------------------------|:-----------|:---------|:--------------|-----------:|---------:|:---------|-----------:|---------:|---------:|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|----------------:|------------------:|-----------------:|
| ESPORTSTMNT01_2690210 | oe:team:733ebb9dbf22a401c0127a0c80193ca | team       | False    | False         |      40224 |    45960 | LCKC     |      40224 |    45960 |      767 |          40136 |        49931 |          864 |             88 |        -3971 |          -97 |           6 |            12 |            7 |               7 |                22 |                6 |
| ESPORTSTMNT01_2690210 | oe:team:7c64febcd5ccff13dcd035dc6867a00 | team       | True     | True          |      40136 |    49931 | LCKC     |      40136 |    49931 |      864 |          40224 |        45960 |          767 |            -88 |         3971 |           97 |           7 |            22 |            6 |               6 |                12 |                7 |
| ESPORTSTMNT01_2690219 | oe:team:731b7a9fd004cdbe2bcb3da795bce47 | team       | False    | False         |      39335 |    49409 | LCKC     |      39335 |    49409 |      895 |          46615 |        57155 |          928 |          -7280 |        -7746 |          -33 |           1 |             1 |            8 |               8 |                13 |                1 |
| ESPORTSTMNT01_2690219 | oe:team:e7a7c6bf58eb268ed3f13aac4158aa8 | team       | True     | True          |      46615 |    57155 | LCKC     |      46615 |    57155 |      928 |          39335 |        49409 |          895 |           7280 |         7746 |           33 |           8 |            13 |            1 |               1 |                 1 |                8 |
| 8401-8401_game_1      | oe:team:f4c4528c6981e104a11ea7548630c23 | team       | True     | False         |        nan |      nan | LPL      |        nan |      nan |      nan |            nan |          nan |          nan |            nan |          nan |          nan |         nan |           nan |          nan |             nan |               nan |              nan |

### Univariate Analysis Plot
### Bivariate Analysis Plot
### Interesting Aggregates

## Assessment of Missingness
### NMAR Analysis
After analyzing the dataset aggregated by teams, we found the `pick1` column to be NMAR due to its inability to be predicted from other columns, and the missingness potentially occuring due to a variable not included in the dataset. Unlike many of the columns we analyzed, the missingness of `pick1` did not depend on league, as many leagues didn't report certain variables as a whole. Instead, 31 out of the 55 leagues had missingness for `pick1` specifically. The missingness also did not depend on `teamid`, as 381 out of 593 teams had a missing value in `pick1`. Instead, the column's missingness depends on the value itself - the champion chosen. Missingness in this column can occur due to the champion being a newly added character to the game, resulting in a null value being recorded. Additional data that would make this column MAR would be data about whether the champion is a newly added character at the time of the game. 
### Missingness Dependency	
Once we realized that metrics 25 minutes into the game were extremely relevant to our research, we wanted to test which columns could influence their missingness. For our test, we compared the impact of the `gamelength` and `firstdragon` columns on the missingness of the `goldat25` column. We performed two permutation tests based on the following two hypothesis pairs:

`gamelength`:
* **Null Hypothesis:** The missingness of `goldat25` is independent of `gamelength`.
* **Alternate Hypothesis:** The missingness of `goldat25` is associated with `gamelength`.

Our results showed a very low p-value, leading us to reject our null hypothesis. Based on the test, it is likely that the missingness of `goldat25` was associated with `gamelength`. This makes sense logically as games that finish early will be more likely to have unrecorded values past their finish time. 

`firstdragon`: 
* Null Hypothesis: The missingness of `goldat25` is independent of `firstdragon`.
* Alternate Hypothesis: The missingness of `goldat25` is associated with `firstdragon`.

Our results returned an insignificant p-value, meaning that we fail to reject the null hypothesis and the missingness of `goldat25` is not influenced by the `firstdragon` column.  

## Hypothesis Testing
* **Null Hypothesis:** The win rate between teams who secure the first dragon is equal to the win rate of teams that do not secure the first dragon.
* **Alternate Hypothesis:** The win rate between teams who secure the first dragon is different from the win rate of teams that do not secure the first dragon.
* **Test statistic:** Difference in win rate between games with and without securing the first dragon. 

Note: testing at significance level of 0.05.

The permutation test resulted in a p-value of 0.0 which is less than 0.05, we reject the null hypothesis. Based on the data, the win rate of teams who secure the first dragon is significantly higher than teams that do not secure the first dragon. 

Since the results were so extreme, (p-value near 0), we can use this in our progress of determining the best circumstances for a win. This test was helpful in determining that `firstdragon` is in fact a very useful feature to consider when predicting win rates. 


## Framing a Prediction Problem
Our prediction model is a classifier to predict the game result using the data at the 25 minute mark. Since the response variable could have one of two results - winning and losing - our model is a binary classifier. Our response variable of choice alligns with all our previous intentions so far to find the most ideal 

## Baseline Model
## Final Model
## Fairness Analysis