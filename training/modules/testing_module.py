class test():
	model = None
	generator = None
	frequencies = None
	df = None
	y_pred = None
	y_true = None
	y_pred_prob = None
	thresholds = None

    #test = testing_module.test("past_models/weights.11-val_loss0.561-val_prec0.650-val_rec0.616.hdf5","train_facs.csv",filter_aus=target_aus,improve_thresholds=False,first_df_elements=1000,images_dir = "/ground0/facs_processed_images",skip_prediction=True)

	def __init__(self,path_to_model,path_to_df,filter_aus=None,improve_thresholds=False,first_df_elements=None,images_dir=None,skip_prediction=False):
		import numpy as np
		from keras.preprocessing import image
		from keras_vggface.vggface import VGGFace
		from keras_vggface import utils
		from matplotlib import pyplot as plt
		from tensorflow.keras.models import load_model
		from keras import backend as K
		import operator
		import pandas as pd

		#load model-----------
		def recall(y_true, y_pred):
			"""Recall metric.

			Only computes a batch-wise average of recall.

			Computes the recall, a metric for multi-label classification of
			how many relevant items are selected.
			"""
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
			recall = true_positives / (possible_positives + K.epsilon())
			return recall

		def precision(y_true, y_pred):
			"""Precision metric.

			Only computes a batch-wise average of precision.

			Computes the precision, a metric for multi-label classification of
			how many selected items are relevant.
			"""
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
			precision = true_positives / (predicted_positives + K.epsilon())
			return precision

		dependencies = {
			 'recall': recall,
			 'precision': precision,
		}

		self.model = load_model(path_to_model, custom_objects=dependencies)
		
		
		#load dataframe-----------
		def _split(x):
						if type(x) == float: #if math.isnan(x):
								return []
						return x.split(",")
		
		df = pd.read_csv(path_to_df)
		df["actionUnits"]=df["actionUnits"].apply(lambda x:_split(x))
		if first_df_elements is not None:
			df = df[:first_df_elements]
		self.df = df
		
		if filter_aus == None:
			self.calculateFrequencies()
		else:
			self.filterAus(filter_aus)
			self.calculateFrequencies()
			
		self.makeGenerator(images_dir)
		self.evaluateModel(improve_thresholds=improve_thresholds,skip_prediction=skip_prediction)
		
		
#		 aus_at_index = [1,10,12,14,15,17,2,23,24,4,6,7]
#		  thresholds = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

#		  #AUS DICTIONARY
#		  aus_dic = {0:"Neutral face",1:"Inner brow raiser",2:"Outer brow raiser",4:"Brow lowerer",5:"Upper lid raiser",6:"Cheek raiser",7:"Lid tightener",8:"Lips toward each other",9:"Nose wrinkler",10:"Upper lip raiser",11:"Nasolabial deepener",12:"Lip corner puller",13:"Sharp lip puller",14:"Dimpler,buccinator",15:"Lip corner depressor",16:"Lower lip depressor",17:"Chin raiser",18:"Lip pucker",19:"Tongue show",20:"Lip stretcher",21:"Neck tightener",22:"Lip funneler",23:"Lip tightener",24:"Lip pressor",25:"Lips part",26:"Jaw drop",27:"Mouth stretch",28:"Lip suck"}
#		  index_description = list(map(lambda x: aus_dic[x],aus_at_index))
#		  print(index_description)


	def makeGenerator(self,images_dir):
		from keras.preprocessing.image import ImageDataGenerator
		from keras_vggface import utils

		
		datagen=ImageDataGenerator()
		
		train_generator=datagen.flow_from_dataframe(
			dataframe=self.df,
			directory=images_dir,
			x_col='filePath',
			y_col='actionUnits',
			batch_size=1,
			shuffle=False,
			class_mode="categorical",
			target_size=(224,224),
		)
		
		self.generator = train_generator
		
	def calculateFrequencies(self):
		import operator
		
		df = self.df
			
		all_aus = set()
		frequency = {}
		total=0
		for aus in df["actionUnits"]:
			for au in aus:
				total+=1
				all_aus.add(au)
				if(au in frequency):
					frequency[au]=frequency[au]+1
				else:
					frequency[au]=1   
		sorted_frequencies = sorted(frequency.items(), key=operator.itemgetter(1),reverse=True)
		

		self.frequencies = sorted_frequencies

			
	def filterAus(self,target_aus):
		import pandas as pd
		df = self.df.values

		reduced_df = []
		for i in range(len(df)):
			aus = df[i][1]
			new_aus=[]
			for au in aus:
				if au in target_aus:
					new_aus.append(au)
			if len(new_aus)!=0:
				reduced_df.append([df[i][0],new_aus])

		reduced_df = pd.DataFrame(reduced_df,columns = ['filePath' , 'actionUnits']) 
		self.df = reduced_df
		self.calculateFrequencies()
		

	def evaluateModel(self,improve_thresholds=False,skip_prediction = False):
		print("Starting testing...")
		import numpy as np
		y_pred_prob=self.y_pred_prob
		if not skip_prediction:
			y_pred_prob = self.model.predict_generator(self.generator,workers=0)
			self.y_pred_prob = y_pred_prob
		y_true = np.array(self.returnY_true(self.generator.labels,12))
		self.y_true = y_true
		thresholds = None
		if improve_thresholds:
			thresholds = self.findThresholds(y_pred_prob,y_true)
			self.thresholds = thresholds
		print("Found thresholds are:")
		print(thresholds)
		y_pred = self.binarize_predictions(y_pred_prob,thresholds)
		self.y_pred = y_pred
		self.printMetrics("Results",y_pred,y_true)

	def printMetrics(self,title,pred,true):    
		from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score, mean_squared_error
		print("\n{}---------------------------------------------".format(title))
		print("exact match ratio: {}".format(accuracy_score(pred,true)))
		print("mse: {}".format(mean_squared_error(pred,true)))
		print("micro precision: {}".format(precision_score(pred,true,average='micro')))
		print("macro precision: {}".format(precision_score(pred,true,average='macro')))
		print("micro recall: {}".format(recall_score(pred,true,average='macro')))
		print("macro recall: {}".format(recall_score(pred,true,average='macro')))
		print("micro f1-score: {}".format(f1_score(pred,true,average='micro'))) #Micro is good for label imbalance --seems like a decent one
		print("macro f1-score: {}".format(f1_score(pred,true,average='macro'))) #Does not take label imbalance into account
		print("weighted f1-score: {}".format(f1_score(pred,true,average='weighted'))) #Like macro but accounts for label imbalance
		print("hamming-loss: {}".format(hamming_loss(pred,true))) #fractions of labels incorrectly predicted 

	## returns a multiple hot encoded vector of aus predictions given a vector of probabilities and the individual thresholds.
	def binarize_predictions(self,predictions,thresholds):
		import numpy as np
		if thresholds is None:
			thresholds = np.ones(len(predictions)) / 2
		binary_predictions = np.zeros(predictions.shape)

		for p in range(len(predictions)):
			for au in range(len(predictions[p])):
				if predictions[p][au]>thresholds[au]:
					binary_predictions[p][au]=1
				else:
					binary_predictions[p][au]=0
		return binary_predictions

	def findThresholds(self,y_pred_prob,y_true):
		import numpy as np
		false_values=np.zeros(len(y_true[0]))
		count_false = np.zeros(len(y_true[0]))
		true_values=np.zeros(len(y_true[0]))
		count_true = np.zeros(len(y_true[0]))
		for t in range(len(y_true)):
			for i in range(len(y_true[t])):
				if y_true[t][i] ==0:

					false_values[i]+= y_pred_prob[t][i]
					count_false[i]+=1
				else:
					true_values[i]+= y_pred_prob[t][i]
					count_true[i]+=1
		false_values = np.divide(false_values,count_false)
		true_values = np.divide(true_values,count_true)

		thresholds = false_values + (true_values - false_values)/2
		return(thresholds)

	def returnY_true(self,y_labels,number_of_labels):
		import numpy as np
		y_true = []
		for label in y_labels:
			new = np.zeros(number_of_labels)
			for l in label:
				new[l]=1
			y_true.append(new)
		return y_true


