from sklearn.svm import LinearSVC
import json
import numpy as np
from tqdm import tqdm

json_file = "outputs_stylegan/stylegan2/scores_stylegan2.json"
with open(json_file) as json_data:
    attributes_dict = json.load(json_data)

print(len(attributes_dict['logits']))

np_file = "outputs_stylegan/stylegan2/w.npy"
w = np.load(np_file)

print(w.shape)

X = w
y = np.round(np.array(attributes_dict['logits']))

print(X.shape)
print(y.shape)
print(y)

list_SVM = []

for i in tqdm(range(40)):
    clf = LinearSVC(random_state=0, tol=1e-5).fit(X, y[:,i])
    list_SVM.append(clf)
    break
breakpoint() # TODO get the boundary