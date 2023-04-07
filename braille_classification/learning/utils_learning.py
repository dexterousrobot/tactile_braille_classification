import numpy as np
import torch


class LabelEncoder:

    def __init__(self, label_names, device='cuda'):
        self.device = device
        self.target_label_names = label_names

    @property
    def out_dim(self):
        return len(self.target_label_names)

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
