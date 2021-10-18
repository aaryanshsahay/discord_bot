############################################# LIBRARIES ####################################################################

import discord
import time
from datetime import datetime
#Machine-Learning Libraries
import numpy as np
import matplotlib.pyplot as plt
# Linear Regression
from sklearn.linear_model import LinearRegression
# from sklearn.feature_extraction import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# Neural Networks
from keras.models import model_from_json
from keras import models
import cv2
#Movie Recommendation

#Misc
import urllib.request


bot=discord.Client()
bot_key=''

############################################# BOT EVENTS ###################################################################

@bot.event
async def on_ready():
	print('Logged in as {0.user}'.format(bot))

@bot.event
async def on_message(message):

	# Hi/Hello/Hey message
	if message.content.startswith('!hello') or message.content.startswith('!hi') or message.content.startswith('!hey'):
		msg='Hello there {}, Glad to meet you! Use !help to get started quickly.'.format(message.author)
		await message.channel.send(msg)
		# Logging
		print('Gretting message sent to {}'.format(message.author))

	# Main Help Message
	if message.content.startswith('!help'):
		msg=help_main
		await message.channel.send(msg)

	# Linear Regression Command
	if message.content.startswith('!linreg x'):
		global x
		x=message.content.split('x')
		x=x[1].strip()
		x=x.split(',')
		x=[int(i) for i in x]
		x=np.asarray(x)
		x=x.reshape(x.size,1)
		msg='X Input taken Successfully!'
		await message.channel.send(msg)

	if message.content.startswith('!linreg y'):
		global y
		y=message.content.split('y')
		y=y[1].strip()
		y=y.split(',')
		y=[int(i) for i in y]
		y=np.asarray(y)
		y=y.reshape(y.size,1)
		msg='Y Input taken Successfully!'
		await message.channel.send(msg)

	if message.content.startswith('!linreg predict'):
		z=message.content.split('predict')
		z=int(z[1])
		z=np.asarray(z)
		z=z.reshape(z.size,1)
		await message.channel.send('Prediction value taken successfully')
		
		#Calling Linear Regression Function.
		linear_regression=lin_reg(x,y,z)
		score,prediction,slope,intercept=linear_regression['Score'],linear_regression['Prediction'],linear_regression['Slope'],linear_regression['Intercept']
		preds='Prediction is : {}'.format(prediction[0][0])
		await message.channel.send(preds)
		
		#Plotting Graph.
		graph=get_graph(x,y,slope,intercept)
		graph[0].figure.savefig(r'Images/1234.jpg')
		graph[0].figure.clf()
		await message.channel.send(file=discord.File(r'Images/1234.jpg'))

		await message.channel.send('The score for this model is : {}'.format(score))

	# Linear Regression Help Command
	if message.content.startswith('!linreg help'):
		msg=help_linreg
		await message.channel.send(msg)

	# Dog Cat Classsifier Command
	if message.content.startswith('!dogcat'):
		
		query=message.content.split('!dogcat')
		query=query[1]
		urllib.request.urlretrieve(query,'dogcat.jpg')

		model=load_model(json_dogcat,weights_dogcat)

		pred=dogcat(model,'dogcat.jpg')

		msg='Wow Its a {}'.format(pred)

		await message.channel.send(msg)

	# Dog Cat Classifier Help Command
	if message.content.startswith('!dogcat help'):
		msg=help_dogcat
		await message.channel.send(msg)

	# Equation Solver Command
	if message.content.startswith('!solve'):
		pass

	# Equation Solver Help Command
	if message.content.startswith('!solve help'):
		msg=help_solve
		await message.channel.send(msg)

	# Recommendation command
	if message.content.startswith('!recommend'):
		pass

	# Recommendation Help Command
	if message.content.startswith('!recommend help'):
		msg=help_recommend
		await message.channel.send(msg)

	# Personality Command
	if message.content.startswith('!personality'):
		pass


	# Personality Help Command
	if message.content.startswith('!personality help'):
		msg=help_personality
		await message.channel.send(msg)

############################################# FUNCTIONS ##################################################################
def lin_reg(x,y,z):
    try:
        reg=LinearRegression()
        reg=reg.fit(x,y)
        predict=reg.predict(z)

        slope=reg.coef_
        intercept=reg.intercept_

        score=reg.score(x,y)

        output={
        'Score':score,
        'Prediction':predict,
        'Slope':slope,
        'Intercept':intercept
        }

        return output
    except:
        error_='something went wrong try again later'
        return error_

def get_graph(x,y,slope,intercept):
	x1=np.linspace(0,10,100).reshape(100,1)
	y1=slope*x1+intercept
	graph=plt.scatter(x,y)
	graph=plt.plot(x1,y1,color='r')
	
	return graph

def load_model(json_file,weights):
	json_file=open(json_file,'r')
	loaded_model_json=json_file.read()
	json_file.close()

	loaded_model=models.model_from_json(loaded_model_json)
	loaded_model.load_weights(weights)

	return loaded_model

def dogcat(model,image):
	img=cv2.imread(image)
	if img is not None:
		grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		resized_img=cv2.resize(grey_img,(64,64))
	final_img=resized_img.reshape(1,(64*64))

	pred=model.predict(final_img)

	if pred==1:
		res='Cat'
	else:
		res='Dog'

	return res


def movie_recommendation():
    df = pd.read_csv() #add file location#
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df["combined_features"] = df.apply(combined_features, axis =1)  
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)
    movie_user_likes = input("Enter your choice of movie:\n")
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[int(movie_index)]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
    i=0
    print(f"Recommended Movies to watch if you like \"{movie_user_likes}\":\n")
    for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i=i+1
        if i>3:
            break
    
def combined_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]



############################################# FILE NAMES  ################################################################

json_dogcat='model.json'
weights_dogcat='catdog.h5'


############################################# MESSAGES ###################################################################

help_main='''Thanks For using Lux Bot, Here are a list of commands to help you get started! If youd like to learn more about the command use [!command help].\n
1)!linreg : takes input (x and y and x1) from user and returns the predicted value y1 for x1 and the best fit line.
2)!dogcat : takes an image as an input and predicts whether the image is of a dog or a cat.
3)!solve : takes an image as an input which contains a simple mathematical equation and returns the result.
4)!recommend : takes a name of a movie as an input and returns 3 titles similiar to it.
5)!personality : describe yourself and based on the input the bot will predict your personality type.
'''

help_linreg='''This command trains a Linear Regression Model. How to use it:\n
1) use !linreg x [integers] to input the x values. For Example !linreg x 1,2,3,4.\n
2) use !linreg y [integers] to input y values. For Example !linreg y 2,4,6,8. The number of elements must be the exact same as x.\n
3) use !linreg predict [integer] to get the predicted value for an integer based on the data inserted above.\n
4) This will now return the predicted value along with the graph for the best fit line and display the score for the model. '''

help_dogcat='''This command takes input as a link for a jpg image and the classifies whether the image is that of a dog, or a cat.\n
Some Information about the Model:\n
An Artificial Neural Netowork was trained on 2000 images (1000 for cat and 1000 for dog)\n->Testing accuracy (on 80-20 split) is 60%.  '''

help_solve=''' ''' 

help_recommend=''' '''

help_personality=''' '''

bot.run(bot_key) 	
