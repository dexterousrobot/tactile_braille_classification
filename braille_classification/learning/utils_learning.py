import numpy as np
import torch


ARROW_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
ALPHA_NAMES = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
               'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
               'Z', 'X', 'C', 'V', 'B', 'N', 'M',
               'SPACE', 'NONE']


def csv_row_to_label(row):
    return {
        'id': np.array(row['obj_id'] - 1),
        'label': row['obj_lbl'],
    }


class LabelEncoder:

    def __init__(self, out_dim, target_label_names, device):
        self.device = device
        self.out_dim = out_dim
        self.target_label_names = target_label_names

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.

        Returns: torch tensor that will be predicted by the NN
        """
        return torch.nn.functional.one_hot(labels_dict['id'], num_classes=self.out_dim).float().to(self.device)

    def decode_label(self, outputs):
        """
        Process NN predictions to raw pose data, always decodes to cpu.

        Returns: Dict of np arrays in suitable format for downstream task.
        """

        ids = outputs.argmax(dim=1).detach().cpu().numpy()
        return {
            'id': ids,
            'label': np.array(self.target_label_names)[ids]
        }
