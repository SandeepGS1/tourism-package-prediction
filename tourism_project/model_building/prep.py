
import pandas as pd

df = pd.read_csv("hf://datasets/SandeepGS/tourism_package-prediction/tourism.csv")

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

Xtrain = X.iloc[:3000]
Xtest = X.iloc[3000:]

ytrain = y.iloc[:3000]
ytest = y.iloc[3000:]

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Prep done")
