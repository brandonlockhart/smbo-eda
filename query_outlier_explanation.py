from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, STATUS_FAIL, Trials
import numpy as np
import pandas as pd
from numpy.random import normal

np.random.seed(1)

# length of dataframe
N = 20000
# mean of normal distribution
mu = 80
# variance of normal distribution
sigma = 10
# number of values per dimension
n = 100
# outlier category
outlier = 0
# scale for denominator
c = 0.9


def data():
    synth = {"a1": [], "a2": [], "ad": [], "av": []}
    for i in range(N):
        a1 = np.random.choice(range(n))
        a2 = np.random.choice(range(n))
        ad = i % 10
        if ad < 5:
            if (a1 >= 40 and a1 <= 60) and (a2 >= 40 and a2 <= 60):
                av = normal(loc=mu, scale=sigma)
            # elif (a1 >= 20 and a1 <= 80) and (a2 >= 20 and a2 <= 80):
            #     av = normal(loc=(mu + 10) / 2, scale=sigma)
            else:
                av = normal(loc=10, scale=sigma)
        else:
            av = normal(loc=10, scale=sigma)
        synth["a1"].append(a1)
        synth["a2"].append(a2)
        synth["ad"].append(ad)
        synth["av"].append(av)
    return pd.DataFrame(synth)


# define an objective function
def objective(args):
    if args["a1_lower"] >= args["a1_upper"] or args["a2_lower"] >= args["a2_upper"]:
        return {"status": STATUS_FAIL}
    df_temp = df[
        ~(
            (df["a1"] >= args["a1_lower"])
            & (df["a1"] <= args["a1_upper"])
            & (df["a2"] >= args["a2_lower"])
            & (df["a2"] <= args["a2_upper"])
        )
    ]
    agg_remove = df_temp.groupby("ad").sum().iloc[outlier]["av"]
    print(args)
    # print(agg_remove, -(agg - agg_remove) / (len(df) - len(df_temp)))

    # df_temp2 = df[
    #     ~((df["a1"] >= 40) & (df["a1"] <= 60) & (df["a2"] >= 40) & (df["a2"] <= 60))
    # ]
    # agg_remove2 = df_temp2.groupby("ad").sum().iloc[outlier]["av"]
    # print(agg_remove2, -(agg - agg_remove2) / ((len(df) - len(df_temp2)) ** c))

    # df_temp3 = df[
    #     ~((df["a1"] >= 35) & (df["a1"] <= 65) & (df["a2"] >= 35) & (df["a2"] <= 65))
    # ]
    # agg_remove3 = df_temp3.groupby("ad").sum().iloc[outlier]["av"]
    # print(agg_remove3, -(agg - agg_remove3) / ((len(df) - len(df_temp3)) ** c))

    # df_temp4 = df[
    #     ~((df["a1"] >= 45) & (df["a1"] <= 55) & (df["a2"] >= 45) & (df["a2"] <= 55))
    # ]
    # agg_remove4 = df_temp4.groupby("ad").sum().iloc[outlier]["av"]
    # print(agg_remove4, -(agg - agg_remove4) / ((len(df) - len(df_temp4)) ** c))

    return {
        "loss": -(agg - agg_remove) / ((len(df) - len(df_temp)) ** c),
        "status": STATUS_OK,
    }


df = data()
df = df[df["ad"] == outlier]
agg = df.groupby("ad").sum().iloc[outlier]["av"]

# define a search space
a1_min = min(list(df["a1"].unique()))
a1_max = max(list(df["a1"].unique()))
a2_min = min(list(df["a2"].unique()))
a2_max = max(list(df["a2"].unique()))
q = 1

space = hp.choice(
    "parameters",
    [
        {
            "a1_lower": hp.quniform("a1_lower", a1_min, a1_max, q),
            "a1_upper": hp.quniform("a1_upper", a1_min, a1_max, q),
            "a2_lower": hp.quniform("a2_lower", a2_min, a2_max, q),
            "a2_upper": hp.quniform("a2_upper", a2_min, a2_max, q),
        },
    ],
)

trials = Trials()
# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, trials=trials)
print(space_eval(space, best))
