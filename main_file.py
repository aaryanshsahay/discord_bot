######################################## LIBRARIES ##################################################

import discord
import time
from datetime import datetime
#Machine-Learning Libraries
import numpy as np
import matplotlib.pyplot as plt
# Linear Regression
from sklearn.linear_model import LinearRegression
# Neural Networks
from keras.models import model_from_json
from keras import models
import cv2
#Movie Recommendation
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Personality
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
#nltk.download('stopwords')
#import nltk
#nltk.download('wordnet')
import gzip
#Misc
import urllib.request
import pickle
import re
import joblib


bot=discord.Client()
bot_key='ODM0NzQ2MzkzODY3NTgzNDg5.YIFYKQ.DqBbwzqMxcLLJZfXZ6GJgehQI4Y'

######################################################################################################
######################################## BOT EVENTS ##################################################
######################################################################################################

@bot.event
async def on_ready():
	print('Logged in as {}'.format(bot.user))

@bot.event
async def on_message(message):

	# Ignore messages sent by bot
	if message.author==bot.user:
		return

	# Greeting 
	if message.content.startswith('!hello') or message.content.startswith('!hi') or message.content.startswith('!hey'):
		embed=discord.Embed(description=greeting_message.format(message.author))
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Help command
	if message.content.startswith('!help'):
		embed=discord.Embed(title='Help Center',description=help_main)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Linear Regression Command
	if message.content.startswith('!linreg x'):
		try:
			global x
			x=message.content.split('x')
			x=x[1].strip()
			x=x.split(',')
			x=[int(i) for i in x]
			x=np.asarray(x)
			x=x.reshape(x.size,1)
			embed=discord.Embed(description='X Input taken Successfully!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)
		except:
			embed=discord.Embed(title='Error',description='The model only takes in integers! Try again!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)


	if message.content.startswith('!linreg y'):
		try:
			global y
			y=message.content.split('y')
			y=y[1].strip()
			y=y.split(',')
			y=[int(i) for i in y]
			y=np.asarray(y)
			y=y.reshape(y.size,1)
			embed=discord.Embed(description='Y Input taken Successfully!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)
		except:
			embed=discord.Embed(title='Error',description='The model can only take integers, try again!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)
	
	if message.content.startswith('!linreg predict'):
		z=message.content.split('predict')
		z=int(z[1])
		z=np.asarray(z)
		z=z.reshape(z.size,1)

		embed=discord.Embed(description='Prediction value taken successfully! Calculating now...')
		embed.color=0x0a528c
		await message.channel.send(embed=embed)
		
		#Calling Linear Regression Function.
		linear_regression=lin_reg(x,y,z)
		score,prediction,slope,intercept=linear_regression['Score'],linear_regression['Prediction'],linear_regression['Slope'],linear_regression['Intercept']
		preds='Prediction is : {}'.format(prediction[0][0])
		
		
		#Plotting Graph.
		graph=get_graph(x,y,slope,intercept)
		graph[0].figure.savefig(r'Images/1234.jpg')
		graph[0].figure.clf()

		embed=discord.Embed(title='Linear Regression',description='The linear regression model trained on :')
		embed.color=0x0a528c
		embed.add_field(name='X values :',value=x.tolist(),inline=True)
		embed.add_field(name='Y values :',value=y.tolist(),inline=True)
		embed.add_field(name=f'Prediction On {z.tolist()[0][0]}',value=prediction[0][0],inline=False)
		file = discord.File("Images/1234.jpg", filename="image1.jpg")
		embed.set_image(url="attachment://image1.jpg")
		await message.channel.send(file=file,embed=embed)

	# Linear Regression Help Command
	if message.content.startswith('!linreg help'):
		embed=discord.Embed(title='Linear Regression',description=help_linreg)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Cat Dog Classification
	if message.content.startswith('!dogcat p'):
		try:
			query=message.content.split('!dogcat p ')
			query=query[1]
			urllib.request.urlretrieve(query,'dogcat.jpg')

			model=load_model(json_dogcat,weights_dogcat)

			pred=dogcat(model,'dogcat.jpg')

			embed=discord.Embed(title='Dog Cat Classification',description='The model predicts the image to be a {}'.format(pred))
		
			file = discord.File("dogcat.jpg", filename="image2.jpg")
			embed.set_image(url="attachment://image2.jpg")
			embed.color=0x0a528c
			await message.channel.send(file=file,embed=embed)
		except:
			embed=discord.Embed(title='Error',description='Something went wrong, try a different image!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)

	# Cat Dog Classification Help Command
	if message.content.startswith('!dogcat help'):
		embed=discord.Embed(title='Dog Cat Classification',description=help_dogcat)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Movie recommendation 
	if message.content.startswith('!recommend m'):
		try:
			query=message.content.split('!recommend m ')
			query=query[1]
			df=load_data_for_movie_rec(movie_data)['df']
			cosine_sim=load_data_for_movie_rec(movie_data)['cosine_sim']

			res=get_recommendation(df,query,cosine_sim)
			
			
			embed=discord.Embed(title='Movie recommendation' ,description='Here are some movies which are similiar to {}:'.format(query))
			embed.add_field(name=res['movie1'],value=res['about1'],inline=False)
			embed.add_field(name=res['movie2'],value=res['about2'],inline=False)
			embed.add_field(name=res['movie3'],value=res['about3'],inline=False)
			embed.color=0x0a528c

			await message.channel.send(embed=embed)
		except:
			embed=discord.Embed(title='Error',description='Something went wrong, Make sure you are entering the correct spelling! if the error persists, try some other movie!')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)

	# Movie Recommendation Help command
	if message.content.startswith('!recommend help'):
		embed=discord.Embed(title='Movie Recommendation',description=help_movie)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Insult Identifier command
	if message.content.startswith('!insult i'):
		query=message.content.split('!insult i ')
		query=query[1]
		loaded_model=pickle.load(open(insult_classif, 'rb'))
		get_insult=hs_model(loaded_model,query)
		embed=discord.Embed(title='Insult Identifier',description=f'Woah that was {get_insult}')
		embed.color=0x0a528c
		await message.channel.send(embed=embed)

	# Insult Identifier Help Command
	if message.content.startswith('!insult help'):
		embed=discord.Embed(title='Insult',description=help_insult)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)


	# Personality Classifier Command
	if message.content.startswith('!personality p'):
		try:
			query=message.content.split('!personality p ')
			query=query[1]
			loaded_model=joblib.load(open(personality_model,'rb'))
		

			print(query)
			cntizer=CountVectorizer(analyzer='word',max_features=1000,max_df=0.7,min_df=0.1)
			tfizer=TfidfTransformer()

			personality_type=predict_personality(loaded_model,query,cntizer,tfizer)

			embed=discord.Embed(title='MBTI Personality',description=f'You are most probably a {personality_type}')
			embed.color=0x0a528c
			await message.channel.send(embed=embed)
		except:
			embed=discord.Embed(title='Error',description='This command is currently offline, the team is working to fix the bug!')
			embed.color=0x0a528c
			file = discord.File("Images/error_1.jpg", filename="image3.jpg")
			embed.set_image(url="attachment://image3.jpg")
			await message.channel.send(file=file,embed=embed)

	# Personality Classifier Help Command
	if message.content.startswith('!personality help'):
		embed=discord.Embed(title='Personaliity Prediction',description=help_personality)
		embed.color=0x0a528c
		await message.channel.send(embed=embed)




######################################################################################################
######################################## FUNCTIONS ###################################################
######################################################################################################

def lin_reg(x,y,z):
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
		res='Dog'
	else:
		res='Cat'

	return res

def get_index_from_title(df,title):
	return df[df.title==title]['index'].values[0]

def get_title_from_index(df,index):
	return df[df.index==index]['title'].values[0]

def get_overview_from_index(df,index):
	return df[df.index==index]['overview'].values[0]

def load_data_for_movie_rec(file_path):
	df=pd.read_csv(file_path)
	cv=CountVectorizer()
	count_matrix=cv.fit_transform(df['combined_features'])
	cosine_sim=cosine_similarity(count_matrix)

	return {
		'df':df,
		'cosine_sim':cosine_sim
	}

def get_recommendation(df,user_input,cosine_sim):
	user_input=user_input.replace('[^\w\s]','').lower()
	movie_index=get_index_from_title(df,user_input)
	similar_movies=list(enumerate(cosine_sim[int(movie_index)]))
	sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

	movie_name_1=get_title_from_index(df,sorted_similar_movies[1][0])
	movie_name_2=get_title_from_index(df,sorted_similar_movies[2][0])
	movie_name_3=get_title_from_index(df,sorted_similar_movies[3][0])

	about_movie_1=get_overview_from_index(df,sorted_similar_movies[1][0])
	about_movie_2=get_overview_from_index(df,sorted_similar_movies[2][0])
	about_movie_3=get_overview_from_index(df,sorted_similar_movies[3][0])

	return {
		'movie1':movie_name_1,
		'movie2':movie_name_2,
		'movie3':movie_name_3,
		'about1':about_movie_1,
		'about2':about_movie_2,
		'about3':about_movie_3
		}

def process_text():
	return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

def hs_model(loaded_model,query):
	inp = pd.Series(query)
	yhat = ((np.ravel(loaded_model.predict(inp)).tolist()))
	res=''
	if yhat[-1] == 1:
		res='an insult :rage:'
	else:
		res='not an insult :smiling_face_with_3_hearts:'
	return res

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

def translate_personality(b_Pers,personality):
	
    return [b_Pers[l] for l in personality]

    # transform mbti to binary vector

def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
  lemmatiser = WordNetLemmatizer()
  useless_words = stopwords.words("english")
  unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
  unique_type_list = [x.lower() for x in unique_type_list]
  
  list_personality = []
  list_posts = []
  len_data = len(data)
  i=0
  
  for row in data.iterrows():
      # check code working 
      # i+=1
      # if (i % 500 == 0 or i == 1 or i == len_data):
      #     print("%s of %s rows" % (i, len_data))

      #Remove and clean comments
      posts = row[1].posts

      #Remove url links 
      temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

      #Remove Non-words - keep only words
      temp = re.sub("[^a-zA-Z]", " ", temp)

      # Remove spaces > 1
      temp = re.sub(' +', ' ', temp).lower()

      #Remove multiple letter repeating words
      temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

      #Remove stop words
      if remove_stop_words:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
      else:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
          
      #Remove MBTI personality words from posts
      if remove_mbti_profiles:
          for t in unique_type_list:
              temp = temp.replace(t,"")

      # transform mbti to binary vector
      type_labelized = translate_personality(b_Pers,row[1].type) #or use lab_encoder.transform([row[1].type])[0]
      list_personality.append(type_labelized)
      # the cleaned data temp is passed here
      list_posts.append(temp)

  # returns the result
  list_posts = np.array(list_posts)
  list_personality = np.array(list_personality)
  return list_posts, list_personality

def predict_personality(model,query,cntizer,tfizer):
	mydata=pd.DataFrame(data={'type':['INFJ'],'posts':[query]})

	query,dummy=pre_process_text(mydata,remove_stop_words=True,remove_mbti_profiles=True)
	query=cntizer.fit_transform(query)
	query=tfizer.fit_transform(query).toarray()
	query=model.predict(query)
	return translate_back(query)

######################################################################################################
######################################## FILE PATH ###################################################
######################################################################################################
json_dogcat='models/model.json'
weights_dogcat='models/catdog.h5'

movie_data='data/dataset.csv'


insult_classif='models/HS.pkl'

personality_model='models/personality_model_.pkl'
######################################################################################################
######################################## MESSAGES ####################################################
######################################################################################################

greeting_message='Hello there {} :wave: Glad to meet you! Use !help to get started quickly.'

help_main='''
Welcome to the help-desk for our Bot - **LUX**, this is a basic guide for you to get started. 
All the possible commands are highlighted.  use `!command help` to get more information on a particular command. Have Fun :)

  :chart_with_downwards_trend:__** LINEAR REGRESSION  **__:chart_with_upwards_trend:

> `!linreg x ` , `!linreg y` , `!linreg predict` , `!linreg help`
> 
> - linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables.
> - Our model returns a graph with the best fit line and also makes a prediction for you!
> - Use the last command to get an example.

  __**:cat2:  DOG CAT CLASSIFICATION  :dog2:**__

> `!dogcat p` , `!dogcat help`
> 
> - Using Neural Network, we can classify between 2 images, being dogs and cats. Input an image and let the bot work it's magic! 
> - Use the second command to get an example. 

  __**:projector: MOVIE RECOMMENDATION  :movie_camera:**__

> `!recommend m` , `!recommend help`
> 
> - Have you ever been confused about what movie to watch next, you really liked a movie and want to watch more like these, but google failed you, don't mind that we are here to save you!
> - Use the second command to get an example.

  __**:person_golfing: PERSONALITY :person_doing_cartwheel:**__

> `!personality p` , `!personality help`
> 
> - Use the first command and then write an essay describing yourself and the bot will predict your personality type based on 
>  the MBTI (Myers-Briggs) Personality Types! 
> - Use the second command to get an example.

  :angry:__**  INSULT **__:smiling_face_with_3_hearts: 

> `!insult i` , `!insult help`
> 
> - Use the first command and then write a short sentence.
> - Use the second command to get an example.

~~TADA!!!!~~
'''

help_linreg='''
> This command trains a Linear Regression Model. How to use it:

1) Use `!linreg x [integers]` to input the x values. For Example `!linreg x 1,2,3,4`.

2) Use `!linreg y [integers]` to input y values. For Example `!linreg y 2,4,6,8`. The number of elements must be the exact same as x.

3) Use `!linreg predict [integer]` to get the predicted value for an integer based on the data inserted above.

4) This will now return the predicted value along with the graph for the best fit line and display the score for the model.
'''

help_dogcat='''
> This command takes input as a link for a jpg image and the classifies whether the image is that of a dog, or a cat.
> Some Information about the Model:
 
 1) Use `!dogcat p [link]` for example `!dogcat p https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/02/322868_1100-800x825.jpg` 
 
 2) An Artificial Neural Netowork was trained on 2000 images (1000 for cat and 1000 for dog) 
 ->Testing accuracy (on 80-20 split) is 60%.
'''

help_movie='''
> This command recommends you 3 movies along with a brief overview.
 
 1) Use `!recommend m [movie name]` for example `!recommend m cars`.
 
 2) Cosine similarity was used to find similar titles.
'''

help_insult='''
> This command is used to check if a message you sent is an insult or not.

1) You only have to use `!insult i [message]` to check is your message is insulting to other users or not.

2) Make sure you use "i" (lower case i)

3) `!insult i @user you are an idiot`.
'''

help_personality='''
> Use this command to enter a paragraph about yourself and get your personality type.

1) Use `!personality p [query]` where query will be the paragraph. An example of a query would be

> Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity.

2) This model uses NLP and XGBoost.

'''
bot.run(bot_key)
