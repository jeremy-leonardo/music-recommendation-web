import pandas as pd
import json
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def bulkTrain(arrayOfIndexInput = [], arrayOfOutput = []):
	
	global mlp

	print('\n-- TRAIN --\n')

	# for i in range(len(arrayOfOutput)):
	# 	if arrayOfOutput[i] > 1:
	# 		arrayOfOutput[i] = arrayOfOutput[i]/4

	print('INPUT:')
	print(arrayOfIndexInput)
	print('OUTPUT:')
	print(arrayOfOutput)

	X = data.take(arrayOfIndexInput, axis = 0) 
	X = X.take([15, 28, 30, 34], axis = 1) 
	print('INPUT DF:')
	print(X)

	y = arrayOfOutput

	mlp.fit(X, y)


def bulkPredict(arrayOfIndexInput = []):

	global mlp

	print('\n-- PREDICT --\n')

	# for i in range(len(arrayOfOutput)):
	# 	if arrayOfOutput[i] > 1:
	# 		arrayOfOutput[i] = arrayOfOutput[i]/4

	print('INPUT:')
	print(arrayOfIndexInput)

	X = data.take(arrayOfIndexInput, axis = 0) 
	X = X.take([15, 28, 30, 34], axis = 1) 
	print('INPUT DF:')
	print(X)
	predictions = mlp.predict(X)

	print('\nPREDICTION RESULT:')
	print(predictions)
	print()

	# print(confusion_matrix(y,predictions))
	# print(classification_report(y_test,predictions))
	# print(classification_report(y,predictions))	


def exportDataframe(dataframe, filename = 'export', format = 'csv'):

	filename = filename + '.' + format
	
	dataframe.to_csv ('exported/'+filename, index = None, header=True)


def saveModel(model, filename = 'model', format = 'sav'):
	filename = filename + '.' + format
	pickle.dump(model, open('exported/'+filename, 'wb'))


def loadModel(file):
	loaded_model = pickle.load(open(file, 'rb'))
	return loaded_model



# ---- MAIN ----

data_dir = "./fma_metadata_modified/tracks.csv"
data = pd.read_csv(data_dir,low_memory=False)
data = data.truncate(after=999)

for i in range(1000):
	if data.iloc[i].track_genres_all == '[]' :
		data.iloc[i].track_genres_all = '[-1]'
	if data.iloc[i].track_genres == '[]' :
		data.iloc[i].track_genres = '[-1]'
	data.iloc[i].track_genres_all = json.loads(data.iloc[i].track_genres_all)[0]
	data.iloc[i].track_genres = json.loads(data.iloc[i].track_genres)[0]
	data.iloc[i].album_favorites = int(data.iloc[i].album_favorites)
	data.iloc[i].album_listens = int(data.iloc[i].album_listens)
	data.iloc[i].artist_favorites = int(data.iloc[i].artist_favorites)
	data.iloc[i].track_favorites = int(data.iloc[i].track_favorites)
	data.iloc[i].track_interest = int(data.iloc[i].track_interest)
	data.iloc[i].track_listens = int(data.iloc[i].track_listens)
	# print(data.iloc[i].track_id)

print('DATA:')
print(data)

mlp = MLPClassifier(hidden_layer_sizes=(3), shuffle=False, random_state=42)

bulkTrain(arrayOfIndexInput = [1,2,12,361,494,433,533,201], arrayOfOutput = [2,2,2,2,2,0,0,0])
bulkPredict(arrayOfIndexInput = [208,437,3])

# saveModel(mlp)