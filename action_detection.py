import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns

class RoofBolter:
    """
    Roofbolter machine class
    """

    def __init__(self, machine_name):
        self.machine_name = machine_name

    def load_data_from_cvs_file(self, file_name):
        path_dir = os.getcwd()
        path_file = os.path.join(path_dir, 'Data', file_name + ".csv")
        if os.path.exists(path_file):
            dataset = pd.read_csv(path_file)
            return dataset
        else:
            print("Unable to find the file at {file_path}")


    def prepare_data_for_model(self, processed_data):
        """
        The prepare_data_for_model transforms data into correct formats of data to build a model:
        splits data into training set and test set , normalizing data and encode class lable
        :param processed_data: dataset is preprocessed
        :return: training set and test set
        """
        # set X and y variable to the values of the features and label
        X = processed_data.drop('activity', axis=1).values
        y = processed_data['activity'].values

        #sprit data for training model: 70% training, 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y, test_size=0.30,
                                                random_state=90)
        #Normalizing data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # encode string label to integer
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoder.fit(y_test)
        encoded_y_train = encoder.transform(y_train)
        encoded_y_test = encoder.transform(y_test)

        # convert integers to dummy variables
        dummy_y_train = utils.to_categorical(encoded_y_train)
        dummy_y_test = utils.to_categorical(encoded_y_test)
        return X_train, X_test, dummy_y_train, dummy_y_test


    def train_ANNs_model(self, X_train, X_test, y_train, y_test):
        """
        Train a Artificial Neuron Networks-ANNs.
        The model has three layers: a input, a hidden , a ouput.
        Numer of neuron in first layer = number of feature.
        Numer of neuron in hidden layer = a half of input layer.
        Numer of neuron in output layer = number of classes.
        From the first layers, relu action function is apply to learn model.
        Softmax function is used in the last layer to classify multiclasses.
        :param X_train: set of features to train model
        :param X_test:  set of features to test model
        :param y_train: set of label to train model
        :param y_test: set of label to test model
        :return: a model
        """

        num_features = X_train.shape[1]         #16 features
        num_neuron_hidden_layer = int(num_features/2)
        num_uneuron_output = 7         #number of classes label in dataset

        #Generate model
        model = Sequential()
        #input layer
        model.add(Dense(num_features, activation='relu'))
        # Hidden layer
        model.add(Dense(num_neuron_hidden_layer, activation='relu'))
        # Output layer
        model.add(Dense(num_uneuron_output, activation='softmax'))

        #compile model with loss 'categorical_crossentropy' for multiclass
        #optimizing model with Adam method, Evaluate with accuracy metric
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        #stop training model to avoid overfitting after 25 epoch if no improvment model
        early_stop = EarlyStopping(monitor='val_loss', mode='min',
                                   verbose=1, patience=25)

        # Fit model with 400 epochs and 70000 samples
        print('Starting trainning Artificial Neuron Networks- ANNs model')
        model.fit(x=X_train,
                  y=y_train,
                  epochs=400,
                  batch_size=70000,
                  validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stop]
                  )

        model.save('Roofbolter_ANNs_model.h5')
        print('Model is saved as : Roofbolter_ANNs_model.h5')
        return model

    def get_sample_to_predict(self, file_name, dataset_model):
        """
        Thí function takes randomly a sample from sensor dataset for prediction action
        :param file_name: name of sensor data file
        :param dataset_model: training dataset
        :return: a sample
        """

        #read original data file
        dataset_reproduce = self.load_data_from_cvs_file(file_name)
        #delete missing data
        dataset_reproduce = dataset_reproduce.drop(['hydraulic_pump', 'timestamp']
                                                   , axis=1)
        # delete NAN value
        dataset_reproduce = dataset_reproduce.dropna()
        # delete index number in data
        dataset_reproduce = dataset_reproduce.loc[:, \
                            ~dataset_reproduce.columns.str.contains('^Unnamed')]

        # Get a random sample from reproduce dataset
        #random.seed(90)
        rand_idx = random.randint(0, len(dataset_reproduce))
        sample = dataset_reproduce.iloc[rand_idx]
        # print("colums: ", dataset_reproduce.columns)
        print('\n\n Sample from draw data of sensor: \n sample', sample)
        sample_X = dataset_reproduce.drop('activity', axis=1).iloc[rand_idx].values
        sample_y = dataset_reproduce['activity'].iloc[rand_idx]
        sample = (sample_X, sample_y)
        return sample



    def printout_prediction(self, result):
        print('==================================================================\n\n')
        if result != -1:
            print("      CORRECT PREDICTION")
            print('Action of Roofbolter: ', result )
        else:
            print(print("      INCORRECT PREDICTION"))

        print('\n\n==================================================================')


    def prediction_action(self, model, dataset_train_model, sample):
        """
        The prediction_action predicts the status of machine
        :param model: model of data
        :return: Status of machine
        """
        sample_X, sample_y = sample

        # set X and y variable to the values of the features and label
        X = dataset_train_model.drop('activity', axis=1).values
        y = dataset_train_model['activity'].values

        # Normalizing data
        scaler = MinMaxScaler()
        scaler.fit_transform(X)

        # get class name from encode class of training data
        encoder = LabelEncoder()
        encoder.fit(y)
        class_name = encoder.classes_

        #Take sample from X_transform
        num_feature = len(dataset_train_model.columns) - 1
        sample_tranform_X = scaler.transform(sample_X.reshape(1, num_feature))

        #predict the class for sample
        predict_label_idx = model.predict_classes(sample_tranform_X)
        print('predict idx', predict_label_idx)
        print('classes name in data set', class_name)

        #check prediction and label of sample
        predict_label = class_name[predict_label_idx]
        if (predict_label == sample_y):
            return predict_label
        else:
            return -1


    def evaluation_model(self, model, X_test, y_test):
        predictions = model.predict_classes(X_test)
        class_name = ['Anchoring', 'Drilling', 'Hole Setup', 'Idle', 'Machine Off',
                      'Transitional Delay', 'Traveling']
        print(classification_report(y_test, predictions, target_names = class_name))
        print(multilabel_confusion_matrix(y_test, predictions, labels= class_name))
        losses = pd.DataFrame(model.history.history)
        #losses[['loss', 'val_loss']].plot()
        losses.plot()
        plt.show()


def main():
    #create RoofBolter machine
    roof_bolter = RoofBolter('Machine Roofbolter')
    dataset_train_model = roof_bolter.load_data_from_cvs_file('data_train_model')

    #print dataset
    print('columns:', dataset_train_model.columns)
    print(dataset_train_model.head(5))

    #Tranform data into training set and test set from df_training
    X_train, X_test, y_train, y_test = roof_bolter.prepare_data_for_model(dataset_train_model)

    #train model or load model
    model = roof_bolter.train_ANNs_model(X_train, X_test, y_train, y_test)
    #model = load_model('Roofbolter_ANNs_model.h5')

    #evaluation
    roof_bolter.evaluation_model(model, X_test, y_test)

    #take a random sample
    for i in range(10):
        #selet sample from dataset
        sample = roof_bolter.get_sample_to_predict('data_case_study', dataset_train_model)
        #prediction state from sensor data
        prediction = roof_bolter.prediction_action(model, dataset_train_model, sample)
        roof_bolter.printout_prediction(prediction)


if __name__ == "__main__":
     main()


