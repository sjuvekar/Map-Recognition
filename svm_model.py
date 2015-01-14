import numpy
from sklearn import svm, preprocessing
from model import Model

class SvmModel(Model):

    def __init__(self, X, y):
        Model.__init__(self, X, y)
        self.classifier = svm.SVC(gamma=1000, verbose=3)

    def preprocess(self):
        self.X = numpy.log(self.X.abs()) * numpy.sign(self.X)
        Model.preprocess(self)
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
