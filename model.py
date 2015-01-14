import numpy
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class Model(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classifier = None

    def preprocess(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2, random_state=1)
    
    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def test(self):
        return self.classifier.score(self.X_test, self.y_test)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        
    def model_stats(self):
        y_pred = self.classifier.predict(self.X_test)
        labels = list(sorted(set(self.y)))
        cm = confusion_matrix(self.y_test.astype(str), y_pred.astype(str), labels=labels)
        fig = plt.figure(figsize=(8,8))

        ax1=fig.add_subplot(111)
        ax1.set_frame_on(False)

        plt.matshow(numpy.log(cm), 1)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.xticks(xrange(0, len(labels)), labels, rotation=90, fontsize="small")
        plt.yticks(xrange(0, len(labels)), labels,fontsize="small")

        plt.show()

        print "Precision of the models is: ",
        print precision_score(self.y_test, y_pred, labels=labels)

        print "Recall of the models is: ",
        print recall_score(self.y_test, y_pred, labels=labels)

        print "F1 score of the models is: ",
        print f1_score(self.y_test, y_pred, labels=labels)

