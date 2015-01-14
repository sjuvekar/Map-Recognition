import numpy
from sklearn.ensemble import GradientBoostingClassifier
from model import Model

class GradientBoostingModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
        self.classifier = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=1,
                                                     subsample=0.6, verbose=3)

    def preprocess(self):
        #self.X = numpy.log(self.X.abs()) * numpy.sign(self.X)
        Model.preprocess(self)

