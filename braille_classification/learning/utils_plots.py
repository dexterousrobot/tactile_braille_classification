import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class ClassErrorPlotter:
    def __init__(
        self,
        class_names,
        save_dir=None,
        name="error_plot.png",
        plot_during_training=False,
        normalize=True
    ):
        self.class_names = class_names
        self.save_dir = save_dir
        self.name = name
        self.plot_during_training = plot_during_training
        self.normalize = normalize

        if plot_during_training:
            plt.ion()
            plt.figure()
            self._fig = plt.gcf()
            self._fig.set_size_inches((12, 12), forward=False)

    def update(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):

        self._fig.gca().clear()

        cm = metrics['conf_mat']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=90, fontsize=12)
        plt.yticks(tick_marks, self.class_names, fontsize=12)

        fmt = '.2f' if self.normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

        plt.tight_layout()
        plt.xlabel('Target class', fontsize=16, fontweight='bold')
        plt.ylabel('Predicted class', fontsize=16, fontweight='bold')

        if self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        if not self.plot_during_training:
            plt.figure()
            self._fig = plt.gcf()
            self._fig.set_size_inches((12, 12), forward=False)

        self.update(pred_df, targ_df, metrics)
        plt.show(block=True)


if __name__ == '__main__':
    pass
