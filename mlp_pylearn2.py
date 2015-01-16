import numpy
import pandas
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
from model import Model
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from theano import tensor as T
import theano

class MLPPylearnModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
      
        train="""
          !obj:pylearn2.train.Train {
            dataset: &train !obj:pylearn2.datasets.csv_dataset.CSVDataset {
              path: 'csv/img_train.csv'
            },
            model: !obj:pylearn2.models.mlp.MLP {
              layers: [
                       !obj:pylearn2.models.mlp.Sigmoid {
                           layer_name: 'h0',
                           dim: 100,
                           irange: .01,
                       },

                       !obj:pylearn2.models.mlp.Softmax {
                           layer_name: 'y',
                           n_classes: 50,
                           irange: 0.
                       }
                      ],
              nvis: 3600,
            },
            algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
              batch_size: 50,
              learning_rate: .2,
              learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                  init_momentum: 0.5,
              },
              monitoring_dataset:
                  {
                      'train' : *train,
                      'valid' : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/img_valid.csv'
                      },
                      'test'  : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/img_test.csv'
                      }
                  },
              termination_criterion: !obj:pylearn2.termination_criteria.And {
                  criteria: [
                      !obj:pylearn2.termination_criteria.EpochCounter {
                          max_epochs: 100
                      }
                  ]
              }
            },
            extensions: [
              !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                   channel_name: 'valid_y_misclass',
                   save_path: "mlp_best.pkl"
              },
            ]
          }
          """
        self.classifier = yaml_parse.load(train)
        self.model_path = "mlp_best.pkl"
        test = pandas.read_csv("csv/img_test.csv")
        self.X_test = test[range(60 * 60)]
        self.y_test = test["State"]


    def preprocess(self):
        pass

    def train(self):
        self.classifier.main_loop()

    def test(self):
        y_pred = self.predict()
        return metrics.accuracy_score(self.y_test, y_pred)

    def predict(self):
        model = serial.load(self.model_path)
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop( X )
        Y = T.argmax( Y, axis = 1 )
        f = theano.function( [X], Y )
        return f(self.X_test)
    
    def confusion_matrix(self, y_pred):
        labels = list(sorted(set(self.y_test)))
        cm = confusion_matrix(self.y_test, y_pred, labels=labels)
        return (cm, labels)
