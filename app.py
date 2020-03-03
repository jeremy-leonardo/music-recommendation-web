from sklearn.neural_network import MLPClassifier
from flask import Flask, redirect, request, url_for
from flask import render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)


def bulkTrain(arrayOfIndexInput = [], arrayOfOutput = []):
	
	mlp = MLPClassifier(hidden_layer_sizes=(3), shuffle=False, random_state=42)

	print('INPUT TRAIN:')
	print(arrayOfIndexInput)
	print('OUTPUT TRAIN:')
	print(arrayOfOutput)

	data = getTracks()

	X = data.take(arrayOfIndexInput, axis = 0) 
	X = X.take([15, 28, 30, 34], axis = 1) 
	print('INPUT DF:')
	print(X)

	y = arrayOfOutput
	mlp.fit(X, y)
	return mlp



def bulkPredict(mlp, arrayOfIndexInput = []):

	print('INPUT PREDICT:')
	print(arrayOfIndexInput)

	data = getTracks()

	X = data.take(arrayOfIndexInput, axis = 0) 
	X = X.take([15, 28, 30, 34], axis = 1) 
	print('INPUT PREDICT DF:')
	print(X)
	predictions = mlp.predict(X)

	print('\nPREDICTION RESULT:')
	print(predictions)
	return predictions



def getRecommendationArrayOfIndex():
	X = loadUserInput()
	y = loadUserOutput()
	mlp = bulkTrain(X,y)

	fullIndexArray = []
	for i in range(1000):
		fullIndexArray.append(i)

	filteredArrayOfIndex = np.delete(fullIndexArray, X)
	predictionArray = bulkPredict(mlp, filteredArrayOfIndex)

	recommendedArrayOfIndex = []
	for i in range(len(filteredArrayOfIndex)):
		if(predictionArray[i] == 2):
			recommendedArrayOfIndex.append(filteredArrayOfIndex[i])

	return recommendedArrayOfIndex



def getTracks():

	data_dir = "./fma_metadata_modified/tracks.csv"
	data = pd.read_csv(data_dir,low_memory=False)
	data = data.truncate(after=999)

	genres_dir = "./fma_metadata_modified/genres.csv"
	genres = pd.read_csv(genres_dir,low_memory=False)

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
		# data.iloc[i].track_genre_top = genres.iloc[int(data.iloc[i].track_genres) - 1].title

	return data;



def saveUserInput(data = []):
	# with open('saved-user-input.csv', 'w') as file:
	# 	writer = csv.writer(file)
	# 	writer.writerows(data)
	data = json.dumps(data)
	file = open(r"saved-user-input.dat","w")
	print('Saved user input data:')
	print(data)
	file.write(data)
	file.close()



def saveUserOutput(data = []):
	# with open('saved-user-input.csv', 'w') as file:
	# 	writer = csv.writer(file)
	# 	writer.writerows(data)
	data = json.dumps(data)
	file = open(r"saved-user-output.dat","w")
	print('Saved user output data:')
	print(data)
	file.write(data)
	file.close()



def loadUserInput():
	file = open("saved-user-input.dat","r+")  
	data = file.read()
	data = json.loads(data)
	file.close()
	print('Loaded user input data:')
	print(data)
	return data



def loadUserOutput():
	file = open("saved-user-output.dat","r+")  
	data = file.read()
	data = json.loads(data)
	file.close()
	print('Loaded user output data:')
	print(data)
	return data





# -- ROUTES -- #



@app.route('/')
def index():

	data = getTracks()
	data = data.take([0, 36, 15, 28, 30, 34], axis = 1)

	return render_template('index.html', column_names=data.columns.values, row_data=list(data.values.tolist()))



@app.route('/recommendations')
def recommendations():

	data = getTracks()
	recommendation_array_of_index = getRecommendationArrayOfIndex()
	data = data.take(recommendation_array_of_index, axis = 0) 
	data = data.take([0, 36, 15, 28, 30, 34], axis = 1)

	return render_template('index.html', column_names=data.columns.values, row_data=list(data.values.tolist()))



@app.route('/musics-all-columns')
def musicAllColumns():

	data = getTracks()

	return render_template('index.html', tables=[data.to_html(classes='data')], titles=data.columns.values)



@app.route('/like/<track_id>/<track_index>', methods=['POST'])
def like(track_id, track_index):
	if request.method == 'POST':
		print('Liked track id '+ str(track_id))
		print('Liked track index '+ str(track_index))
		tracks = loadUserInput()
		tracks.append(int(track_index))
		saveUserInput(tracks)

		loadedOutput = loadUserOutput()
		loadedOutput.append(2)
		saveUserOutput(loadedOutput)

		# return ('Liked track id '+ str(track_id) +' index '+ str(track_index), 200)
		# return redirect('/', code=200)
		return redirect(url_for('index'))



@app.route('/dislike/<track_id>/<track_index>', methods=['POST'])
def dislike(track_id, track_index):
	if request.method == 'POST':
		print('Disliked track id '+ str(track_id))
		print('Disliked track index '+ str(track_index))
		tracks = loadUserInput()
		tracks.append(int(track_index))
		saveUserInput(tracks)

		loadedOutput = loadUserOutput()
		loadedOutput.append(0)
		saveUserOutput(loadedOutput)

		# return ('Disliked track id '+ str(track_id) +' index '+ str(track_index), 200)
		# return redirect('/', code=200)
		return redirect(url_for('index'))



@app.route('/unsure/<track_id>/<track_index>', methods=['POST'])
def unsure(track_id, track_index):
	if request.method == 'POST':
		print('Unsure about track id '+ str(track_id))
		print('Unsure about track index '+ str(track_index))
		tracks = loadUserInput()
		tracks.append(int(track_index))
		saveUserInput(tracks)

		loadedOutput = loadUserOutput()
		loadedOutput.append(1)
		saveUserOutput(loadedOutput)

		# return ('Unsure about track id '+ str(track_id) +' index '+ str(track_index), 200)
		# return redirect('/', code=200)
		return redirect(url_for('index'))



@app.route('/reset')
def reset():
	print('Reset success!')
	empty_array = []
	saveUserInput(empty_array)
	saveUserOutput(empty_array)

	return 'Reset success!'