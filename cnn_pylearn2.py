import numpy
from sklearn import preprocessing
from model import Model
from pylearn2.config import yaml_parse

class CNNPylearnModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
      
        train="""
          !obj:pylearn2.train.Train {
            dataset: &train !obj:pylearn2.datasets.csv_dataset.CSVDataset {
              path: 'csv/img_train.csv'
            },
            model: !obj:pylearn2.models.mlp.MLP {
              input_space: !obj:pylearn2.space.Conv2DSpace {
                       shape: [60, 60],
                       num_channels: 1
              },
              layers: [
                       !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                           layer_name: 'h2',
                           output_channels: 64,
                           kernel_shape: [5, 5],
                           pool_shape: [4, 4],
                           pool_stride: [2, 2],
                           irange: .05
                       },
                       !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                           layer_name: 'h3',
                           output_channels: 64,
                           kernel_shape: [5, 5],
                           pool_shape: [4, 4],
                           pool_stride: [2, 2],
                           irange: .05
                       },
                       !obj:pylearn2.models.mlp.Softmax {
                           layer_name: 'y',
                           n_classes: 50,
                           irange: 0.
                       }
                      ]
            },
            algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
              batch_size: 100,
              learning_rate: .01,
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
              cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
                  !obj:pylearn2.costs.cost.MethodCost {
                      method: 'cost_from_X'
                  }, !obj:pylearn2.costs.mlp.WeightDecay {
                      coeffs: [ .00005, .00005, .00005 ]
                  }
                ]
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
                   save_path: "cnn_best.pkl"
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
