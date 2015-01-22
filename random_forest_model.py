import numpy
from sklearn.ensemble import RandomForestClassifier
from model import Model

class RandomForestModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=16, verbose=3)

    def preprocess(self):
        #self.X = numpy.log(self.X.abs()) * numpy.sign(self.X)
        Model.preprocess(self)

