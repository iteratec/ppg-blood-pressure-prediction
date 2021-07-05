from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from scipy import stats

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



# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))