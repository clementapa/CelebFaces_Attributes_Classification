import json, os
import numpy as np

from tqdm import tqdm
from sklearn.svm import LinearSVC

from utils.utils_functions import create_dir, balace_input
from utils.constant import ATTRIBUTES

class BoundaryCreator(object):
    def __init__(self, config):
        self.config = config
        output_dir = config.output_dir
        np_file = config.np_file
        json_file = config.json_file
        self.latent_space_dim = config.latent_space_dim

        with open(json_file) as json_data:
            self.attributes_dict = json.load(json_data)
        self.X = np.load(np_file)
        self.raw_y = np.array(self.attributes_dict['logits'])
        self.y = np.round(self.raw_y)
        
        create_dir(output_dir)

    def create_boundaries(self):
        list_boundaries = []
        output_dir = self.config.output_dir

        for i in tqdm(range(40)):
            positive_samples, negative_samples = balace_input(self.X, self.y, i)
            balanced_X = np.concatenate([positive_samples[0], negative_samples[0]], axis=0)
            balanced_Y = np.concatenate([positive_samples[1], negative_samples[1]], axis=0)


            if len(balanced_Y) > 0:
                clf = LinearSVC(random_state=0, tol=1e-5).fit(balanced_X, balanced_Y)
                a = clf.coef_.reshape(1, self.latent_space_dim).astype(np.float32)
                list_boundaries.append(a / np.linalg.norm(a))

                print("{} - Score on the whole set: {}".format(ATTRIBUTES[i], clf.score(self.X, self.y[:, i])))
                print("{} - Accuracy on the train set: {}".format(ATTRIBUTES[i], clf.score(balanced_X, balanced_Y)))
            else:
                print("{} - Not enough samples!".format(ATTRIBUTES[i]))

        
        for i, boundary in enumerate(list_boundaries):
            file = os.path.join(output_dir, 'boundary_{}.npy'.format(ATTRIBUTES[i]))
            np.save(file, boundary)
