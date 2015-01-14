import cv2
import cPickle
import numpy
import itertools
from scipy import misc
import pandas

SCALED_SIZE = (60, 60)

def flatten_image(img):
    return misc.imresize(img, SCALED_SIZE).ravel()
 
def rotate(input_img, angle):
    rot = cv2.getRotationMatrix2D((input_img.shape[0]/2, input_img.shape[1]/2), angle, 1.0)
    #For some reason we have to swtich the output size dimensions here:
    rotated = cv2.warpAffine(input_img, rot, (input_img.shape[1], input_img.shape[0]), borderValue=255)
    return flatten_image(rotated)

def rescale(input_img, scale):
    scaled = cv2.resize(input_img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return flatten_image(scaled)

def translate(input_img, x, y):
    M = numpy.float32([[1,0,x],[0,1,y]])
    #For some reason we have to swtich the output size dimensions here:
    new_shape = (input_img.shape[1] + max(0, x), input_img.shape[0] + max(0, y))
    dst = cv2.warpAffine(input_img, M, new_shape, borderValue=255)
    return flatten_image(dst)

def transform_state(state_name, state_img):
    """
    Rotate, translate and scale a state
    """
    # 1) Rotate
    angles = range(1, 360)
    numpy.random.shuffle(angles)
    angles = angles[0:30]
    rotations = numpy.array( [rotate(state_img, angle) for angle in angles] )
    #rotations["Method"] = "Rotate"
    
    # 2) Scale
    scale_sizes = numpy.random.rand(30) * 4
    scales = numpy.array( [rescale(state_img, s) for s in scale_sizes] )
    #scales["Method"] = "Scale"
    
    # 3) Translate
    trans_distances = range(200)
    numpy.random.shuffle(trans_distances)
    trans_distances = trans_distances[0:50]
    translations_all = itertools.product(trans_distances, trans_distances)
    translations = numpy.array( [translate(state_img, dx, dy) for dx, dy in translations_all] )
    #translations["Method"] = "Translate"

    all_transformations = pandas.DataFrame( numpy.vstack( (rotations, scales, translations) ) )
    all_transformations["State"] = state_name
    return all_transformations

if __name__ == "__main__":
    all_images = {}
    original_images = cPickle.load(open("data/original_images.pickle", "rb"))
    state_names = original_images.keys()
    for state_name in state_names:
        print state_name
        all_images[state_name] = transform_state(state_name, original_images[state_name])

    final_images = pandas.concat(all_images.values())
    with open("data/raw_images.pickle", "wb") as output:
        cPickle.dump(final_images, output)
