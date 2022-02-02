from sklearn.svm import LinearSVC
import json
import numpy as np
from tqdm import tqdm

def balace_input(X, y, attribute_idx):
    pos_idx, neg_idx = np.where(y[:, attribute_idx]==1)[0], np.where(y[:, attribute_idx]==0)[0]
    num_samples = min(len(pos_idx), len(neg_idx))
    if len(pos_idx) >= len(neg_idx):
        positive_samples = X[pos_idx][:num_samples], y[pos_idx, attribute_idx][:num_samples]
        negative_samples = X[neg_idx], y[neg_idx, attribute_idx]
    else:
        positive_samples = X[pos_idx], y[pos_idx, attribute_idx]
        negative_samples = X[neg_idx][:num_samples], y[neg_idx, attribute_idx][:num_samples]
    return positive_samples, negative_samples

json_file = "outputs_stylegan/stylegan2/scores_stylegan2.json"
with open(json_file) as json_data:
    attributes_dict = json.load(json_data)

print(len(attributes_dict['logits']))

np_file = "outputs_stylegan/stylegan2/w.npy"
w = np.load(np_file)

X = w
raw_y = np.array(attributes_dict['logits'])
y = np.round(raw_y)

latent_space_dim = 512
list_boundaries = []

for i in tqdm(range(40)):
    positive_samples, negative_samples = balace_input(X, y, i)
    balanced_X = np.concatenate([positive_samples[0], negative_samples[0]], axis=0)
    balanced_Y = np.concatenate([positive_samples[1], negative_samples[1]], axis=0)
    clf = LinearSVC(random_state=0, tol=1e-5).fit(balanced_X, balanced_Y)
    a = clf.coef_.reshape(1, latent_space_dim).astype(np.float32)
    list_boundaries.append(a / np.linalg.norm(a))

    print("Score on the whole set: {}".format(clf.score(X, y[:, i])))
    print("Accuracy on the train set: {}".format(clf.score(balanced_X, balanced_Y)))
    break

for i, boundary in enumerate(list_boundaries):
    with open(os.path.join('trained_boundaries', 'boundary_{}.npy'.format(i))) as file:
        np.save(file, boundary)
    