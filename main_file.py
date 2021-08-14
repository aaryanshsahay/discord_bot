#Basic
import discord
import time
from datetime import datetime
#Machine-Learning Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#Keys and Passwords
from references import bot_key

bot=discord.Client()


@bot.event
async def on_ready():
	print('Logged in as {0.user}'.format(bot))

@bot.event
async def on_message(message):
	
	#Hello/Hi/Hey - Greeting Message
	if message.content.startswith('!hello') or message.content.startswith('!hi') or message.content.startswith('!hey'):
		await message.channel.send('Hello There {},Glad to meet you! Use !help for more information!'.format(message.author))
		print('Greeting Message sent to {}'.format(message.author))
	
	#Help/Summary- List Of Commands
	if message.content.startswith('!help'):
		msg='''
		Enter Message
		1) Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pharetra sit amet ante gravida aliquam. Morbi tincidunt non felis eu aliquam. 
		2) Nam finibus et quam id feugiat. Quisque ac est blandit, rhoncus nunc et, ornare est.
		3) Cras venenatis faucibus mi vitae dignissim. In hac habitasse platea dictumst. Vivamus tempus a orci tristique tincidunt.
		'''
		await message.channel.send(msg)
		print('!help command used by: {}'.format(message.author))

	########################################LINEAR REGRESSION########################################

	#Linear Regression X input.
	if message.content.startswith('!linreg x'):
		global x
		x=message.content.split('x')
		x=x[1].strip()
		x=x.split(',')
		x=[int(i) for i in x]
		x=np.asarray(x)
		x=x.reshape(x.size,1)
		await message.channel.send('X input taken successfully!')
	
	#Linear Regression Y input.
	if message.content.startswith('!linreg y'):
		global y
		y=message.content.split('y')
		y=y[1].strip()
		y=y.split(',')
		y=[int(i) for i in y]
		y=np.asarray(y)
		y=y.reshape(y.size,1)
		await message.channel.send('Y input taken successfully!')
	
	#Linear Regression prediction input.
	if message.content.startswith('!linreg predict'):
		z=message.content.split('predict')
		z=int(z[1])
		z=np.asarray(z)
		z=z.reshape(z.size,1)
		await message.channel.send('Prediction value taken successfully')
		
		#Calling Linear Regression Function.
		linear_regression=lin_reg(x,y,z)
		score,prediction,slope,intercept=linear_regression['Score'],linear_regression['Prediction'],linear_regression['Slope'],linear_regression['Intercept']
		preds='Prediction is:{}'.format(prediction)
		await message.channel.send(preds)
		
		#Plotting Graph.
		graph=get_graph_linreg(x,y,slope,intercept)
		graph[0].figure.savefig(r'images/1234.jpg')
		graph[0].figure.clf()
		await message.channel.send(file=discord.File(r'images/1234.jpg'))




def lin_reg(x,y,z):
	reg=LinearRegression()
	reg=reg.fit(x,y)
	score=reg.score(x,y)
	prediction=reg.predict(z)
	slope=reg.coef_
	intercept=reg.intercept_
	

	return {
		'Score':score,
		'Prediction':prediction[0],
		'Slope':slope,
		'Intercept':intercept
	}



def get_graph_linreg(x,y,slope,intercept):
	x1=np.linspace(0,10,100).reshape(100,1)
	y1=slope*x1+intercept
	graph=plt.scatter(x,y)
	graph=plt.plot(x1,y1,color='r')
	return graph


bot.run(bot_key)