from libs import train_utils
from libs.common.project_paths import GetPaths


class Trainer:
    def __init__(self,
                 config=None,
                 model=None,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 criterion=None,
                 optimizer=None,
                 device=None,
                 transform=None,
                 logger=None,
                 checkpoint=None,
                 early_stopping=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.transform = transform
        self.logger = logger
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

    def train(self):
        cfg = self.config

        for epoch in range(cfg['TRAIN']['EPOCHS']):
            # train
            train_loss, train_acc = train_utils.share_loop(
                epoch,
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                mode="train"
            )

            # validation
            valid_loss, valid_acc = train_utils.share_loop(
                epoch,
                self.model,
                self.valid_loader,
                self.criterion,
                self.optimizer,
                self.device,
                mode="valid"
            )

            results = [int(epoch), train_loss, valid_loss, train_acc, valid_acc]
            train_utils.print_result(result=results)  # 결과출력
            self.logger.logging(results)  # 로깅
            self.checkpoint(valid_loss, self.model)  # 체크포인트 저장
            self.early_stopping(valid_loss)  # 얼리스탑
            if self.early_stopping.early_stop:
                break
