import pandas as pd

from dataload.transform import ImageTransform
from libs.common.common import split_datasetv2
from libs.core.tester import Tester
from libs.core.trainer import Trainer
from configs.setting import global_setting
from model.model_load import build_model
from model.opt_load import opt_load
from model.loss_load import loss_load
from libs.logger.csv_logger import CSVLogger
from libs.callbacks.early_stopping import EarlyStopping
from libs.callbacks.save_checkpoint import SaveCheckPoint

from dataload.dataloader import build_dataloader


def main(mode='train'):
    # 모델 설정값 로드
    config, device = global_setting('cfg.yaml')

    # Transform
    transform = ImageTransform()

    if mode == 'train':
        # dataframes
        data_path = './data/train.csv'
        train_df, valid_df, test_df = split_datasetv2(data_path,
                                                      target_column='label',
                                                      save=True,
                                                      seed=config['SEED'])

        # 모델 로드
        model = build_model(config, device, mode='train')

        # 데이터 로더
        train_loader = build_dataloader(config=config, df=train_df, transform=transform.train, mode="train")
        valid_loader = build_dataloader(config=config, df=valid_df, transform=transform.valid, mode="valid")

        # loss, optimizer
        criterion = loss_load(config=config, device=device)
        optimizer = opt_load(config=config, model=model)

        # callbacks
        logger = CSVLogger(
            path=config['TRAIN']['LOGGING_SAVE_PATH'],
            sep=config['TRAIN']['LOGGING_SEP']
        )

        checkpoint = SaveCheckPoint(path=config['TRAIN']['MODEL_SAVE_PATH'],
                                    model_name=config['MODEL']['NAME'],
                                    opt_name=config['MODEL']['OPTIMIZER'],
                                    lr=config['MODEL']['LR'],
                                    )

        early_stopping = EarlyStopping(
            patience=config['TRAIN']['EARLYSTOP_PATIENT'], verbose=True
        )

        train = Trainer(config=config,
                        model=model,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device,
                        transform=transform,
                        logger=logger,
                        checkpoint=checkpoint,
                        early_stopping=early_stopping)
        train.train()

    elif mode == 'test':
        test_df = pd.read_csv('data/split/split_test.csv')
        model = build_model(config, device, mode='test')
        test_loader = build_dataloader(config=config,
                                       df=test_df,
                                       transform=transform.valid,
                                       mode='test')

        test = Tester(config=config,
                      model=model,
                      test_loader=test_loader,
                      device=device)

        test.do_test()
    else:
        assert False


if __name__ == '__main__':

    main('test')



