import numpy as np
import pandas as pd
import torch


class LabelEncoder:

    def __init__(self, label_names, device='cuda'):
        self.device = device
        self.target_label_names = label_names
        self.tolerences = np.ones(len(label_names))

    @property
    def out_dim(self):
        return len(self.target_label_names)

    def encode_label(self, labels_dict):
        """
        Process label data to NN friendly label for prediction.

        Returns: torch tensor that will be predicted by the NN
        """
        return torch.nn.functional.one_hot(labels_dict['id'], num_classes=self.out_dim).float().to(self.device)

    def decode_label(self, outputs):
        """
        Process NN predictions to label data, always decodes to cpu.

        Returns: Dict of np arrays in suitable format for downstream task.
        """

        ids = outputs.argmax(dim=1).detach().cpu().numpy()
        return {
            'id': ids,
            'label': np.array(self.target_label_names)[ids]
        }

    def calc_batch_metrics(self, labels, predictions):
        """
        Calculate metrics useful for measuring progress throughout training.

        Returns: dict of metrics
            {
                'metric': np.array()
            }
        """
        err_df = self.err_metric(labels, predictions)
        acc_df = self.acc_metric(err_df)
        return err_df, acc_df

    def err_metric(self, labels, predictions):
        """
        Error metric for classification problem, returns dict of errors.
        """
        err_df = pd.DataFrame(columns=self.target_label_names)
        items = zip(labels['label'], predictions['label'])

        for label_name in self.target_label_names:
            correct = [pred == label_name and lab == label_name for lab, pred in items]
            err_df[label_name] = correct

        return err_df

    def acc_metric(self, err_df):
        """
        Accuracy metric for classification problem, counting the number of predictions within a tolerance.
        """

        batch_size = err_df.shape[0]
        acc_df = pd.DataFrame(columns=[*self.target_label_names, 'overall_acc'])
        overall_correct = np.ones(batch_size, dtype=bool)

        for label_name, tolerence in zip(self.target_label_names, self.tolerences):
            abs_err = err_df[label_name]
            correct = (abs_err < tolerence)

            overall_correct = overall_correct & correct
            acc_df[label_name] = correct.astype(np.float32)

        # count where all predictions are correct for overall accuracy
        acc_df['overall_acc'] = overall_correct.astype(np.float32)

        return acc_df
