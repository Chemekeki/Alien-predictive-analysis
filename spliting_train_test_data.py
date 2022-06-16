train_df = pd.read_csv("Train_data.csv")
validate_df = pd.read_csv("Validation.csv")

train_df.head(5)
train_df.describe()


train_df_X = train_df.drop(["ID", "galaxy", "Well-Being Index"], axis=1)
train_df_y = train_df["Well-Being Index"]
train_df_X
train_df_y


"""fill the empty cells with zero"""
train_df_X = train_df_X.apply(lambda x: x.fillna(0), axis = 0)


"""Convert to array from dataframe"""
X_data = train_df_X.iloc[:, 0:].values
y_data = train_df_y.values


""" Train, Test Split"""
X_train, X_test, y_train, y_test = train_test_split(X_data, train_df_y, test_size=0.20, random_state=42)


"""Train, Test Split"""
X_train, X_test, y_train, y_test = train_test_split(X_data, train_df_y, test_size=0.20, random_state=42)


"""Feature selection"""
from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X_train, y_train, X_test, k=65):
    fs = SelectKBest(score_func=f_regression, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

