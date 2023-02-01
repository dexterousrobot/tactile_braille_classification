import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sns.set_theme(style="darkgrid")


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    save_dir=None,
    name="cnf_mtrx.png",
    show_plot=True
):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((12, 12), forward=False)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=16, fontweight='bold')

    if save_dir is not None:
        save_file = os.path.join(save_dir, name)
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    if show_plot:
        plt.show()


if __name__ == '__main__':
    pass
