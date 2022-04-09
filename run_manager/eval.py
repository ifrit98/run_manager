import time
import pickle
from math import sqrt
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager
from tensorflow.python.framework.ops import Tensor, EagerTensor


# TODO: import this from style file in frontrow project toplevel
# that way we don't violate DRY
#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report

from itertools import cycle
from scipy import interp


def is_tensor(x):
    return x.__class__ in [Tensor, EagerTensor]

def divisors(n):
    large_divisors = []
    for i in range(1, int(sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def load_eval_results(path='./eval_metrics.pickle'):
    with open(path, 'rb') as f:
        ev = pickle.load(f)
    return ev


def generate_roc_curve(y_test, y_pred, labels, model_str):

    n_classes = len(labels)
    y_test = y_test.numpy() if is_tensor(y_test) else y_test
    y_score = y_pred.numpy() if is_tensor(y_pred) else y_pred

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro ROC (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro ROC (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, label, color in zip(range(n_classes), labels, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='{0} ({1:0.2f})'
                ''.format(label, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC: {} model'.format(model_str))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC_{}_categorical.png'.format(model_str))
    plt.clf()
    plt.cla()

    return fpr, tpr, roc_auc


def evaluate_model(model, model_name, test_dataset, labels, aleatoric=False):
    test_ds = test_dataset # assume a tf.data.Dataset object
    if isinstance(test_ds, list): # make sure we get even batches
        batch_size = int(list(divisors(test_ds[0].shape[0]))[1])
        if batch_size > test_ds[0].shape[0] // 4:
            batch_size = 1
        test_ds[0] = np.reshape(
            test_ds[0], [-1, batch_size] + list(test_ds[0].shape[1:]))
        test_ds[1] = np.reshape(
            test_ds[1], [-1, batch_size] + list(test_ds[1].shape[1:]))
        test_ds = zip(test_ds[0], test_ds[1])

    variance, trues, preds = [], [], []
    for i, x in enumerate(test_ds):
        if i % 10 == 0:
            print("Predicting test batch {}".format(i))
        y_pred = model.predict(x[0])
        if aleatoric:
            variance.append(y_pred[0][:, -1])
            y_pred = y_pred[1] # take only softmax class values
        y_true = x[1]
        preds.append(y_pred + 1e-8) # epsilon for numerical stability
        trues.append(y_true)

    print("Done making Predictions!")
    y_pred = tf.squeeze(tf.concat(preds, 0))
    y_true = tf.concat(trues, 0)

    print("Plotting ROC curve")
    fpr, tpr, roc_auc = generate_roc_curve(
        y_true, y_pred, labels, model_name)

    labels_dict = dict(zip(list(range(len(labels))), labels))
    labels = np.asarray(list(labels_dict.values()))
    print("LABELS:", labels)
    print("\nLABELS_DICT:", labels_dict)

    def decode_labels(arr, labels_dict):
        if len(np.asarray(arr).shape) == 1:
            return np.asarray([labels_dict[x] for x in arr])
        raise ValueError("Multiple labels... use different func to decode labels")

    if is_tensor(y_true):
        y_true = y_true.numpy()
    if is_tensor(y_pred):
        y_pred = y_pred.numpy()

    # Classification report (F1 score) and Confusion Matrix
    pred = np.asarray(list(map(lambda x: np.argmax(x), y_pred)))
    true = np.asarray(list(map(lambda x: np.argmax(x), y_true)))
    true = decode_labels(true, labels_dict)
    pred = decode_labels(pred, labels_dict)
    mat = confusion_matrix(true, pred)
    report = classification_report(true, pred)
    print(report)
    print(mat)

    # Heatmap
    plt.clf(); plt.cla();
    heatmap = sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='Blues')
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.yaxis.get_ticklabels(), labels)))
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.xaxis.get_ticklabels(), labels)))
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot.png")
    print("\nFinished evaulating model.")

    eval_metrics = {
        'false_pos_rate': fpr,
        'true_pos_rate': tpr,
        'variance': variance,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_true': y_true,
        'labels': labels,
        'labels_dict': labels_dict,
        'conf_mat': mat,
        'class_report': report
    }
    with open("eval_metrics.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
    with open("eval_metrics.md", "w") as f:
        f.writelines('\n'.join(list(eval_metrics)))

    return eval_metrics
