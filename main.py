import sys
import cPickle
from svm_model import SvmModel
from gradient_boosting_model import GradientBoostingModel
from mlp_model import MLPModel
from mlp_pylearn2 import MLPPylearnModel
from cnn_model import CNNModel
from cnn_pylearn2 import CNNPylearnModel

if __name__ == "__main__":
    model = "SvmModel"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if model not in ["MLPModel", "CNNModel"]:
        items = cPickle.load(open("data/items.pickle", "rb"))
        X = items[range(7)]
    else:
        items = cPickle.load(open("data/raw_images.pickle", "rb"))
        X = items[range(60 * 60)]
    y = items["State"]
    s = globals()[model](X, y)
    s.preprocess()
    s.train()
    print "Clasifier Score: ", s.test()
    s.model_stats()
