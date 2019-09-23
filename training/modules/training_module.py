class dataset():
    
    """
    Reads the dataset and creates class variables for the dataframes, frequencies, and generators
    """
    train_df = None
    val_df = None
    train_frequency = None
    val_frequency = None
    train_generator = None
    val_generator = None
    
    def __init__(self):
        """
        Read data from csv files, balance datasets 
        """
        
        #     TRAINING DATASET---------------------------------------------------------------------
        
        import operator
        import pandas as pd
        import math
        def _split(x):
            if type(x) == float:
                if math.isnan(x):
                    return []
            else:
                return x.split(",")
        train_df = pd.read_csv("train_facs.csv")
        train_df["actionUnits"]=train_df["actionUnits"].apply(lambda x:_split(x))
        
        self.train_df = train_df
        
        self.calculateFrequencies("train")
        
        #     VALIDATION DATASET---------------------------------------------------------------------
        
        val_df = pd.read_csv("val_facs.csv")
        val_df["actionUnits"]=val_df["actionUnits"].apply(lambda x:_split(x))

        self.val_df = val_df
        
         
                #BALANCE VALIDATION DATASET ACCORDING TO TRAIN DATASET
        
#         val_df_list = val_df.values
#         balanced_set = self.rebalanceSet(val_df_list,train_frequency,val_frequency)
#         val_df = pd.DataFrame(balanced_set,columns = ['filePath' , 'actionUnits']) 

        
        
        ##     TRAIN AND VALIDATION GENERATORS---------------------------------------------
    def makeGenerators(self):
        from keras-preprocessing.image import ImageDataGenerator
        from keras_vggface import utils

        
        datagen=ImageDataGenerator(#rotation_range = 10,
        #                            zoom_range = .2,
                                   horizontal_flip = True,
                                   preprocessing_function=utils.preprocess_input
                                  )

        train_generator=datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory="./",
            x_col='filePath',
            y_col='actionUnits',
            batch_size=32,
            shuffle=True,
            class_mode="categorical",
            target_size=(224,224),
        )

        val_generator=datagen.flow_from_dataframe(
            dataframe=self.val_df,
            directory="./",
            x_col='filePath',
            y_col='actionUnits',
            batch_size=32,
            shuffle=True,
            class_mode="categorical",
            target_size=(224,224)
        )
        
        self.train_generator = train_generator
        self.val_generator = val_generator
        
    def calculateFrequencies(self,mode):
        """
            updates frequencies of val and train datasets
        """
        import operator
        
        if mode == "train":
            df = self.train_df
        else:
            df = self.val_df
            
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
        
        if mode == "train":
            self.train_frequency = sorted_frequencies
        else:
            self.val_frequency = sorted_frequencies
            
    def filterAus(self,target_aus):
        """
            filter train and aus so only the ones present in target_aus remain
        """
        import pandas as pd
        df = self.train_df.values

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
        self.train_df = reduced_df
        self.calculateFrequencies("train")
        
        df = self.val_df.values
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
        self.val_df = reduced_df
        self.calculateFrequencies("val")
        
    def balanceValidation(self):
        val_df_list = self.val_df.values
        
        train_freq_dic = dict(self.train_frequency)
        balanced_set = self.rebalanceSet(val_df_list,train_freq_dic,self.val_frequency)
        self.val_df = pd.DataFrame(balanced_set,columns = ['filePath' , 'actionUnits']) 
        self.calculateFrequencies("validation")
        
    def rebalanceSet(self,dataset,target_freq_,sorted_own_freq_):
        """
        rebalances a dataset to match class frequencies given a dic of frequencies for another dataset 
        (or a fabricated dic of frequences)
        """
        import copy
        target_freq = target_freq_.copy()
        sorted_own_freq = sorted_own_freq_.copy()
        
        #calculate the class with the smallest size in proportion of the target_freq
        min_size = float("inf")
        min_popularity = 0
        min_class = "0"
        for s in sorted_own_freq:
            proportion = s[1] / target_freq[s[0]]
            if proportion<min_size:
                min_size = proportion
                min_popularity = s[1]
                min_class = s[0]

        #use the less popular class in comparison to target_freq to resize target frequency dic
        size_difference = target_freq[min_class] / min_popularity
        for t in target_freq:
            target_freq[t]/=size_difference
        
        reduced_dataset = []
        for d in dataset:
            can_add = True
            for au in d[1]:
                can_add&=target_freq[str(au)]>0
            if(can_add):
                reduced_dataset.append(d)
                for au in d[1]:
                    target_freq[str(au)]-=1
        return reduced_dataset


        
class train():
    def __init__(self,train_generator,val_generator):
        from keras.models import Model, load_model
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
        from keras import optimizers, models, applications
        from keras import backend as K
        import tensorflow as tf
        import os
        import math
        from keras.layers import Dense, GlobalMaxPooling2D
        from keras.models import load_model
        from keras_vggface.vggface import VGGFace
        
        base_model = VGGFace(model='vgg16',include_top=False,input_shape=(224, 224, 3))


        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        # x = Dropout(0.3) (x)
        # and a dense layer
        x = Dense(512, activation='relu')(x)
        # x = Dropout(0.3) (x)
        x = Dense(512, activation='relu')(x)
        # x = Dropout(0.3) (x)
        predictions = Dense(len(train_generator.class_indices), activation='sigmoid')(x)

        # model to be trained
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized) and freeze all convolutional ones (VGGFACE ones)
        for layer in base_model.layers:
            layer.trainable = False

        #define custom metrics
        #https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

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

        model.compile(optimizer=optimizers.Adam(lr=0.001),loss="binary_crossentropy",metrics=[precision,recall])

#         model.summary()

        checkpoint = ModelCheckpoint("past_models/weights_{epoch:02d}-val_loss:{val_loss:.3f}.hdf5", monitor='val_loss', verbose=2, save_best_only=False, mode='min')
        tb = TensorBoard(log_dir='./logs', batch_size=train_generator.batch_size, write_graph=True, update_freq='batch')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        csv_logger = CSVLogger('./logs/vgg.csv', append=True)


        # dependencies = {'auc_roc': auc_roc}
        # model = keras.models.load_model('past_models/weights.01-0.149.hdf5', custom_objects=dependencies)



        history = model.fit_generator(train_generator, 
                                      epochs=100, 
                                      verbose=1,
                                      validation_data=val_generator,
                                      validation_steps=math.ceil(val_generator.samples/val_generator.batch_size),
                                      steps_per_epoch=math.ceil(train_generator.samples/train_generator.batch_size),
                                      callbacks=[checkpoint, early, tb, csv_logger])

