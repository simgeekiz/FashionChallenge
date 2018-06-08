import numpy as np

# Computes the confusion matrix given 'many-hot' encoded predictions and labels.
# Rows in the confusion matrix are in the following order: tp, fp, tn, fn.
def confusion_matrix(y_true, y_pred):
	confusion = np.zeros(228,4)

	for i in range(228):
		confusion[0][0] = sum(y_true[i] & y_pred[i])
		confusion[0][1] = sum((1 - y_true[i]) & y_pred)
		confusion[0][2] = sum(1 - (y_true[i] & y_pred[i]))
		confusion[0][3] = sum(y_true[i] & (1 - y_pred))
	
	return confusion
