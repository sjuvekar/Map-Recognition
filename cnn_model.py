import numpy
from sklearn import preprocessing
from model import Model
from lasagne import layers
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.pool import MaxPool2DLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano.tensor.nnet import sigmoid

class CNNModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
      
        self.classifier = NeuralNet(
            layers = [
                ("input", layers.InputLayer),
                ("conv1", Conv2DLayer),
                ("pool1", MaxPool2DLayer),
            		('dropout1', layers.DropoutLayer),
                ("conv2", Conv2DLayer),
                ("pool2", MaxPool2DLayer),
            		('dropout2', layers.DropoutLayer),
		            ("hidden1", layers.DenseLayer),
            		('dropout3', layers.DropoutLayer),
                ("output", layers.DenseLayer)
                ],
            input_shape=(None, 1, 60, 60),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
           
	          dropout1_p=0.1,
      	    dropout2_p=0.2,
  	        dropout3_p=0.3,
 
            hidden1_num_units=100,
            hidden2_num_units=100,
            output_num_units=50,
            output_nonlinearity=sigmoid,

            update_learning_rate=0.01,
            update_momentum=0.9,
            max_epochs=10000,
            verbose=1)


    def preprocess(self):
        # Convert state names into numbers
        unique_states = set(self.y)
        state_dict = dict(zip(unique_states, range(len(unique_states))))
        self.y = map(lambda a: state_dict[a], self.y)
        self.y = numpy.array(self.y, dtype=numpy.int32)
 
        #self.X = numpy.log(self.X.abs()) * numpy.sign(self.X)
        Model.preprocess(self)
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # reshape
        self.X_train = self.X_train.reshape(-1, 1, 60, 60)
        self.X_test  = self.X_test.reshape(-1, 1, 60, 60)

