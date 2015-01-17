import sys
import pandas
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
        items = pandas.read_pickle("data/items.pickle")
        X = items[range(7)]
    else:
        items = pandas.read_pickle("data/raw_images.pickle")
        X = items[range(60 * 60)]
    y = items["State"]
    s = globals()[model](X, y)
    s.preprocess()
    s.train()
    print "Clasifier Score: ", s.test()
    s.model_stats()
