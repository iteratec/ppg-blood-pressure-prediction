from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics


df1 = pd.read_pickle("Datasets/ML Datasets/dataset_part1.pkl")

df2 = pd.read_pickle("Datasets/ML Datasets/dataset_part2.pkl")

df3 = pd.read_pickle("Datasets/ML Datasets/dataset_part3.pkl")

bpdf1 = pd.read_pickle("Datasets/ML Datasets/dataset_bp1.pkl")

bpdf2 = pd.read_pickle("Datasets/ML Datasets/dataset_bp2.pkl")

bpdf3 = pd.read_pickle("Datasets/ML Datasets/dataset_bp3.pkl")

# combine dataframes

df_frames = [df1, df2, df3]

bp_frames = [bpdf1, bpdf2, bpdf3]

# feature dataframe
df = pd.concat(df_frames)

df = df.drop('SBP', axis = 1)
df = df.drop('DBP', axis = 1)

# blood pressure dataframe
bp = pd.concat(bp_frames)
print(bp)

# combine feature and bp dataframe
frames = [df, bp]
dataset = pd.concat([df, bp], axis=1)

print(dataset)

# remove None values
dataset = dataset.dropna()
print(dataset.describe())

#remove outliers
dataset = dataset[(np.abs(stats.zscore(dataset)) < 3).all(axis=1)]
print(dataset.describe())

# get x labels
y = np.array(dataset['SBP'])
print(y)

# remove bp from dataset
dataset = dataset.drop('SBP', axis = 1)
dataset = dataset.drop('DBP', axis = 1)


# Saving feature names for later use
feature_list = list(dataset.columns)
# Convert to numpy array
X = np.array(dataset)


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)


sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)


reg = MLPRegressor(solver='lbfgs',        #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,            # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(20,10),
                   early_stopping=True, 
                   activation='relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=100000, random_state=1).fit(X_trainscaled, y_train)
print(reg)


y_pred=reg.predict(X_testscaled)
print(reg.score(X_trainscaled, y_train))
print(reg.score(X_testscaled, y_test))
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'mmHg.')
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print("R2 score", metrics.r2_score(y_test, y_pred))


