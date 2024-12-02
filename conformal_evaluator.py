import numpy as np
import pandas as pd
import sys
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_data(data_df):
    Y = data_df["label"]
    X = data_df.drop(["filename", "label", "date", "bin"], axis=1)

    columns = list(X.columns)
    rename_dict = {}
    for col in range(len(columns)):
        rename_dict[columns[col]] = col

    X = X.rename(columns=rename_dict)
    return (X, Y)

def training_main(data):
    classifier = RandomForestClassifier(max_depth=15)
    x, y = get_data(data)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0, random_state=42)
    classifier.fit(X_train, Y_train)
    return classifier

def quantile(data, alpha):
    if alpha < 1 and alpha > 0:
        return np.quantile(data, 1 - alpha)


def centroid(data):
    denom = len(data)
    retVal = data.sum().values
    return retVal/denom

def conformity_score(data, centroid):
    keys = data.T.keys()
    retVal = []
    for key in keys:
        retVal.append(np.linalg.norm(data[key].values - centroid))
    return retVal

class conformal_evaluator():
    def __init__(self, alpha_, data_, model_ = None):
        self.alpha = alpha_
        self.X_train, self.X_cal, self.Y_train, self.Y_cal = self.split_data(data_)
        if model_ != None:
            self.model = self.training_main()
        else:
            self.model = model_
        self.q_hat = self.calculate_q_hat()
    
    def split_data(self, data):
        x, y = get_data(data)
        return train_test_split(x, y, test_size=0.2, random_state=42)

    def training_main(self):
        classifier = RandomForestClassifier(max_depth=15)
        classifier.fit(self.X_train, self.Y_train)
        return classifier
    
    def calculate_q_hat(self):
        centroid1 = centroid(self.X_train)
        cp_scores = conformity_score(self.X_cal, centroid1)
        return quantile(cp_scores, self.alpha)




# if __name__ == "__main__":
#     if sys.argv[1] == "quantile_test":
#         alpha = float(sys.argv[2])
#         data = np.random.randint(0, 10, size = 10)



