# Emotion detection application
Real time emotion recognition based on action units using deep neural networks and clustering. 

[Link](https://www.youtube.com/watch?v=8apggYgj4HA) to YouTube video overview of the project.

Trained [Deep convolutional neural network (CNN)](https://mega.nz/#!9wYUFQra!UJ6tMEUWOe917BE-YpCIYAISnTGf8RfXhFLwdsmzeCE) used by the application to identify the action units active in a face. To be included in the same directory as app.py


## Emotions recognized
The application is able to recognize the seven emotions displayed below. The application is theoretically able to recognize a further emotion –content, but it was excluded from the list due to a lack of recall in the inference.

<span>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/neutral.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/happiness.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/sadness.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/surprise.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/fear.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/anger.png"  width="200"/>
<img src="https://raw.githubusercontent.com/pablolluchr/emotion-detection-app/master/preview/disgust.png"  width="200"/>
</span>

## Run emotion detection application
Run
```console
python app.py
```
to start the emotion detection application. Good frontal lighting is needed for optimal results.

## Run convolutional layers visualizer
Run
```console
python live_convolutions.py
```
to get live visualizations of the convolutional layers in the action unit detection CNN.

[Convolutional-visualizations](convolutional-visualizations) contains sample images of the activations in each layer of the CNN too.

## Training files

[Training](training) contains the source code used to preprocess the datasets and train de CNN. The datasets used for this training are not included for legal reasons but they can be requested from the original authors.

### Datasets used
Cohn-Kanade AU-Coded Facial Expression dataset
* Kanade, T., Cohn, J. F., \& Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
* Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., \& Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

MMI facial expression database collected by Valstar and Panticand
* M.F. Valstar, M. Pantic, “Induced Disgust, Happiness and Surprise: an Addition to the MMI Facial Expression Database”, Proceedings of the International Language Resources and Evaluation Conference, Malta, May 2010
* M. Pantic, M.F. Valstar, R. Rademaker and L. Maat, “Web­based database for facial expression analysis”, Proc. IEEE Int'l Conf. on Multimedia and Expo (ICME'05),Amsterdam, The Netherlands, July 2005

Denver Intensity of Spontaneous Facial Action Database
* Mavadati, S.M.; Mahoor, M.H.; Bartlett, K.; Trinh, P.; Cohn, J.F., "DISFA:A Spontaneous Facial Action Intensity Database," Affective Computing,IEEE Transactions on , vol.4, no.2, pp.151,160, April-June 2013 , doi:10.1109/T-AFFC.2013.4
* Mavadati, S.M.; Mahoor, M.H.; Bartlett, K.; Trinh, P., "Automatic detection of non-posed facial action units," Image Processing (ICIP), 2012 19th IEEE International Conference on , vol., no., pp.1817,1820, Sept. 30 2012-Oct. 3 2012 , doi: 10.1109/ICIP.2012.6467235

BP4D-Spontaneous
* Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu, and Jeff Girard, “BP4D-Spontaneous: A high resolution spontaneous 3D dynamic facial expression database”, Image and Vision Computing, 32 (2014), pp. 692-706  (special issue of the Best of FG13)
* Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, and Peng Liu, “A high resolution spontaneous 3D dynamic facial expression database”, The 10th IEEE International Conference on Automatic Face and Gesture Recognition (FG13),  April, 2013. 
