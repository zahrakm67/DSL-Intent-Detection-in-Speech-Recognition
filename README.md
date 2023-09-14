# DSL-Intent-Detection-in-Speech-Recognition
# Problem Overview
Intent detection in audio files is a classification problem in machine learning that can be very useful in developing audio assistants and audio-based applications. In this kind of problem, audio files are taken from users, and then the class
of intention is recognized by machine learning methods. Here is the problem of detecting intent in the audio files dataset, which is categorized into 7 classes. The dataset consists of a collection of audio files in a WAV format, and each record is
characterized by several attributes. 
# Dataset
The dataset is divided into two parts:
- **Development dataset**: it contains 9854 records in the form of a CSV file, which includes various features. The most important features are the path of the audio file (audio files in the root folder), the type of action required through the intent (Action), and the device involved by intent (Object).
- **Evaluation dataset**: it contains 1455 records in the form of a CSV file, and its features are the same as the development datasets, only with the difference that we do not have access to its labels. Therefore, the proposed model and machine learning algorithms should classify the intent of each record.

In this study, the final intention label can be calculated using action and object attributes.Therefore, the values of these two attributes are combined and create a new label for each record. In the development data set, this combination has created 7 new classes.

Name and the number of Classes in the audio files dataset

| Class Name | Class Encode | Number of Sample |
|------------|--------------|------------------|
| ActivateMusic | 0   |791|
| ChangeLanguagenone |1   |1113|
| DeactivateLights | 2  |552|
| DecreaseHeat | 3   |1189|
| DecreaseVolume | 4   |2386|
| IncreaseHeat | 5   |1209|
| IncreaseVolume | 6   |2614|

# Proposed Approach
1. **Data preprocessing**
    - Labelencoder : according to Table that mentioned above and with the help of the LabelEncoder method, all class names are converted into a numeric type.
    - Trimming silence:  silence has no relevant information, so it can remove from the start and the end of each audio file and makes the algorithms focus on the parts of the audio signal that contain more information
    - Equalization of Scale: machine learning algorithms and deep algorithms require the same length of features;therefore, the length of all audio signals should be the same.
    - Normalization :data normalization is one of the crucial techniques of data preprocessing before the training stage of different methods.
      In this study, the Standard Scaler method has been used.
    
    - Feature Extraction : approximately 768 features are extracted based on the average, maximum, and minimum rows values.
2. **Model selection**
    - 1D-CNN-Intent : The convolutional neural network is of more interest in this study because it can automatically extract new features from the entire audio signal data.
    - Support vector machines (SVM) : Support vector machine is a supervised machine learning algorithm.
    - Random Forest(RF) :  Random Forest is a supervised machine learning algorithm.
    - Decision Tree (DT) : Decision Tree is a supervised machine learning algorithm.
3. **Hyperparameters tuning**
    
Machine learning algorithms have some parameters that can play an important role in the output of audio signal intent classification.
According to Table , a number of values have been considered for each parameter, and eventually, the best value has been selected according to the F1 score.

Various methods have been presented to optimize or select the best parameters. In this study, we used the GridSearchCV which is based on machine learning methods. 

<table>
  <tr> 
  <td> Models </td>
  <td> Hyperparameters </td>
  <td> Values </td>
  <td> Best selected</td>
</tr>
  <tr>
    <td rowspan="2">SVM</td>
    <td >Kernel </td>    
    <td > rbf,poly  </td>
    <td > rbf  </td>
  </tr>
   <tr>
    <td >C </td>    
    <td > 0.01,0.1,10,15,100   </td>
    <td > 10  </td>
  </tr>

   <tr>
    <td rowspan="2">DT</td>
    <td >criterion  </td>    
    <td > entropy,gini  </td>
    <td > entropy  </td>
  </tr>
   <tr>
    <td >min-samples-split </td>    
    <td > 2,4,8,16,32   </td>
    <td > 32  </td>
  </tr>
  
   <tr>
    <td rowspan="2">RF</td>
    <td >criterion  </td>    
    <td > entropy,gini  </td>
    <td > entropy  </td>
  </tr>
   <tr>
    <td >n-estimators </td>    
    <td > 20,40,80,160,200,300   </td>
    <td > 200  </td>
  </tr>
    
   <tr>
    <td rowspan="4">1D-CNN-Intent</td>
    <td >batch Size  </td>    
    <td > 64,128,512,1024  </td>
    <td > 512  </td>
  </tr>
   <tr>
    <td >optimizer </td>    
    <td > Adam,SGD   </td>
    <td > Adam  </td>
  </tr> 
  <tr>
    <td >learning-rate </td>    
    <td > 0.0001 ,0.001,0.1,0.5 </td>
    <td > 0.0001  </td>
  </tr>
   <tr>
    <td >epochs </td>    
    <td > 100,200,400,500,700   </td>
    <td > 500  </td>
  </tr>
</table>

# Result 
- Deep learning approaches (1DCNN-Intent) successfully extract features for Intent classification (classify up to 87% for accuracy).
- Among the machine learning methods, the **SVM algorithm** with a value of **56%** has performed better for the **F1 score**.
  
| Algorithm |Accuracy |precision |recall |F1 |evaluation|
|-----------|---------|----------|-------|---|----------|
| SVM |0.56 |0.54 |0.58 |0.56 |0.48|
| DT |0.32| 0.32 |0.32 |0.32 |-|
| RF |0.47 |0.41 |0.57 |0.44 |-|
| 1D-CNN-Intent |**0.77** |**0.76** |**0.78** |**0.77** |**0.87**|
