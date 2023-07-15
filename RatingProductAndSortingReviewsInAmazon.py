
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon


import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df = pd.read_csv("amazon_review.csv")
df.head(40)
df.sort_values("total_vote", ascending=False)
avg_rating = df["overall"].mean()
###################################################
# Converting type of related columns to datetime
###################################################
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df["reviewTime"].max()
current_date = pd.to_datetime('2014-12-09 00:00:00')
df["days"] = (current_date - df["reviewTime"]).dt.days

# deviding comments into time zones

df["days"].quantile([0.25, 0.50, 0.75])

df.loc[df["days"] <= 282, "overall"].mean()  # output = 4.69579288
df.loc[(df["days"] > 282) & (df["days"] <= 432), "overall"].mean()  # output = 4.6361406
df.loc[(df["days"] > 432) & (df["days"] <= 602), "overall"].mean()  # output = 4.5716612
df.loc[(df["days"] > 602), "overall"].mean()  # output =4.44625407

# calculating avg-rating accorting to time-based weighted average
df.loc[df["days"] <= 282, "overall"].mean() * 28/100 + \
df.loc[(df["days"] > 282) & (df["days"] <= 432), "overall"].mean() * 26/100 + \
df.loc[(df["days"] > 432) & (df["days"] <= 602), "overall"].mean() * 24/100 + \
df.loc[(df["days"] > 602), "overall"].mean() * 22/100

# out = 4.595593

###################################################
# Finding the 20 reviews of the product that will be shown up on the page
###################################################

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# defining score_pos_neg_diff, score_average_rating ve wilson_lower_bound scored functions
###################################################
def score_pos_neg_diff(yes, no):
    return yes - no

def score_average_rating(yes, no):
    if yes + no == 0:
        return 0
    return yes / (yes + no)

def wilson_lower_bound(yes, no, confidence=0.95):
    """
        Function to provide lower bound of wilson score
        :param pos: No of positive ratings
        :param n: Total number of ratings
        :param confidence: Confidence interval, by default is 95 %
        :return: Wilson Lower bound score
        """
    n = yes + no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# calculating scores for each functions

df["score_pos_neg_diff"]= df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                x["helpful_no"]), axis=1)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df[["overall", "total_vote","helpful_yes", "helpful_no", "score_pos_neg_diff",
    "score_average_rating", "wilson_lower_bound"]].head(20)
##################################################

###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df[["helpful_yes", "helpful_no", "score_average_rating", "wilson_lower_bound"]].\
    sort_values("wilson_lower_bound", ascending=False).head(20)
