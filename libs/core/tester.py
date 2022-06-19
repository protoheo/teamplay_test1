from libs import test_utils
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np


class Tester:
    def __init__(self,
                 config=None,
                 model=None,
                 test_loader=None,
                 device=None,
                 ):
        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def do_test(self):
        total_acc, total_pred, total_label = test_utils.test_loop(
            model=self.model,
            data_loader=self.test_loader,
            device=self.device
        )
        precision, recall, fscore, _ = precision_recall_fscore_support(total_label, total_pred, average='macro')
        print("Total Acc : {}".format(np.mean(total_acc)))
        print(confusion_matrix(total_label, total_pred))

        print("Precision : {} , Recall : {} ".format(precision, recall))

        return total_acc

        # self.logger.logging(results)  # 로깅

