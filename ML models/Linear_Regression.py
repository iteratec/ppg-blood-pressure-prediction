import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
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
features = np.array(dataset)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state=0)


lin_reg_mod = LinearRegression()

lin_reg_mod.fit(X_train, y_train)

print(lin_reg_mod.coef_)

pred = lin_reg_mod.predict(X_test)
errors = abs(pred - y_test)

plt.plot(pred, y_test, 'o')
plt.xlabel('Predicted BP')
plt.ylabel('Measured BP')
plt.show()

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

print('Score: ',  lin_reg_mod.score(X_test, y_test))

print('Mean Absolute Error:', round(np.mean(errors), 2), 'mmHg.')
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

