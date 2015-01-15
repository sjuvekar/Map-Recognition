import numpy
from sklearn import preprocessing
from model import Model
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano.tensor.nnet import sigmoid

class MLPModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
      
        self.classifier = NeuralNet(
            layers = [ # Three layers, one hidden
                ("input", layers.InputLayer),
                ("hidden", layers.DenseLayer),
                ("output", layers.DenseLayer)
                ],
            input_shape=(None, X.shape[1]),
            hidden_num_units=500,
            output_nonlinearity=sigmoid, # Can replace by sigmoid
            output_num_units=50, # 50 states

            #Optimization method
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=1000,
            verbose=1)
        

    def preprocess(self):
       # Convert state names into numbers
        unique_states = set(self.y)
        state_dict = dict(zip(unique_states, range(len(unique_states))))
        self.y = map(lambda a: state_dict[a], self.y)
        self.y = numpy.array(self.y, dtype=numpy.int32)
  
        Model.preprocess(self)
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
