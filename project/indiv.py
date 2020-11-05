# https://piazza.com/class/kemg8hc3toq3td?cid=223 allows the usage of sklearn.
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import csv, cv2, numpy as np, matplotlib.pyplot as plt

# filenames
filenames = []

# X
images = []

# y, where 0 is arabidopsis, 1 is tobacco
image_labels = []

ara_file = csv.reader(open('Ara2013-Canon/Metadata.csv', newline=''), delimiter=',')
ara_count = 0
for row in ara_file:
    filenames.append('Ara2013-Canon/'+row[0] + '_rgb.png')
    image_labels.append(0)

tob_file = csv.reader(open('Tobacco/Metadata.csv', newline=''), delimiter=',')
tob_count = 0
for row in tob_file:
    filenames.append('Tobacco/' + row[0] + '_rgb.png')
    image_labels.append(1)

for file in filenames:
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    f = 400 / min(h, w)
    gray = cv2.resize(gray, None, fx=f, fy=f)
    shift = cv2.xfeatures2d.SIFT_create()
    keypoints = shift.detect(gray)
    _, desc = shift.compute(gray, keypoints)

    sample = np.mean(desc, axis=0)
    images.append(sample)

images = np.array(images)

X_train, X_test, y_train, y_test = train_test_split(images, image_labels, test_size=0.8)

# KNN section
knn = neighbors.KNeighborsClassifier(n_neighbors=5)  # accuracy highest when n_neighbors = 3
knn_model = knn.fit(X_train, y_train)
knn_y_predict = knn_model.predict(X_test)
knn_score = 100 * accuracy_score(y_test, knn_y_predict)
knn_avg_recall = 100 * recall_score(y_test, knn_y_predict, average='macro')
knn_predict_prob = knn_model.predict_proba(X_test).tolist()
knn_probabilities = np.array(knn_predict_prob)[:, 1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probabilities)
knn_auc = auc(knn_fpr, knn_tpr)
print('KNN score: %.4f %%' % knn_score)
print('KNN Recall score: %.4f %%' % knn_avg_recall)
print('KNN AUC: %.4f' % knn_auc)
plt.plot(knn_fpr, knn_tpr, color='blue', lw=1, label='ROC (area = %f)' % knn_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# SVM section
svm_model = svm.SVC(decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)
svm_y_predict = svm_model.predict(X_test)
svm_score = 100 * accuracy_score(y_test, svm_y_predict)
svm_avg_recall = 100 * recall_score(y_test, svm_y_predict, average='macro')
svm_predict_prob = svm_model.predict_proba(X_test)
svm_probabilities = np.array(svm_predict_prob)[:, 1]
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probabilities)
svm_auc = auc(svm_fpr, svm_tpr)
print('SVM score: %.4f %%' % svm_score)
print('SVM Recall score: %.4f %%' % svm_avg_recall)
print('SVM AUC: %.4f' % svm_auc)
plt.plot(svm_fpr, svm_tpr, color='blue', lw=1, label='ROC (area = %f)' % svm_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_y_predict = rf_model.predict(X_test)
rf_score = 100 * accuracy_score(y_test, rf_y_predict)
rf_avg_recall = 100 * recall_score(y_test, rf_y_predict, average='macro')
rf_predict_prob = rf_model.predict_proba(X_test)
rf_probabilities = np.array(rf_predict_prob)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
rf_auc = auc(rf_fpr, rf_tpr)
print('Random Forest score: %.4f %%' % rf_score)
print('Random Forest Recall score: %.4f %%' % rf_avg_recall)
print('Random Forest AUC: %.4f' % rf_auc)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=1, label='ROC (area = %f)' % rf_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()