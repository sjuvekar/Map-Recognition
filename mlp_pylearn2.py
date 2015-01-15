import numpy
from sklearn import preprocessing
from model import Model
from pylearn2.config import yaml_parse

class MLPPylearnModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
      
        train="""
          !obj:pylearn2.train.Train {
            dataset: &train !obj:pylearn2.datasets.csv_dataset.CSVDataset {
              path: 'csv/train.csv'
            },
            model: !obj:pylearn2.models.mlp.MLP {
              layers: [
                       !obj:pylearn2.models.mlp.Sigmoid {
                           layer_name: 'h0',
                           dim: 50,
                           sparse_init: 7,
                       },

                       !obj:pylearn2.models.mlp.Softmax {
                           layer_name: 'y',
                           n_classes: 50,
                           irange: 0.
                       }
                      ],
              nvis: 7,
            },
            algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
              batch_size: 10000,
              line_search_mode: 'exhaustive',
              conjugate: 1,
              updates_per_batch: 10,
              monitoring_dataset:
                  {
                      'train' : *train,
                      'valid' : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/valid.csv'
                      },
                      'test'  : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/test.csv'
                      }
                  },
              termination_criterion: !obj:pylearn2.termination_criteria.And {
                  criteria: [
                      !obj:pylearn2.termination_criteria.MonitorBased {
                          channel_name: "valid_y_misclass"
                      },
                      !obj:pylearn2.termination_criteria.EpochCounter {
                          max_epochs: 10000
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


    def preprocess(self):
        pass

    def train(self):
      self.classifier.main_loop()

    def test(self):
      pass

    def model_stats(self):
      pass
