import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
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
train_features, test_features, train_labels, test_labels = train_test_split(features, y, test_size = 0.25, random_state = 42)

#print shapes
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


estimators = [2600]
rmse_list = []
accuracy = []

for i in range(len(estimators)):

    # Instantiate model with n decision trees
    rf = RandomForestRegressor(n_estimators = estimators[i], random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    rmse = np.sqrt(metrics.mean_squared_error(test_labels, predictions))
    rmse_list.append(rmse)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    acc = 100 - np.mean(mape)
    accuracy.append(acc)


# plot different rsme
'''
plt.plot(rmse_list)
plt.show()
'''
# plot different accuracys
'''
plt.plot(accuracy)
plt.show()
'''

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'mmHg.')
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', rmse_list[0])


print('Accuracy:', round(accuracy[0], 2), '%.')


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]