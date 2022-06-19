from dataload.dataloader import build_dataloader, build_testloader
from dataload.transform import ImageTransform
from libs.core.inference import Inference
from configs.setting import global_setting
from model.model_load import build_model


def main_func(img_path, save=False):
    config, device = global_setting('cfg.yaml')

    # model
    model = build_model(config, device, mode='test')
    transform = ImageTransform()

    inf = Inference(config=config,
                    model=model,
                    transform=transform,
                    device=device)

    return inf.do_inference(img_path, save)


def strfy(x):
    if x == 0:
        ret = '10-2'
    elif x == 10:
        ret = '10-1'
    else:
        ret = str(x)
    return ret


def main_file(file_path):
    import pandas as pd
    config, device = global_setting('cfg.yaml')

    test_df = pd.read_csv(file_path)

    model = build_model(config, device, mode='test')
    transform = ImageTransform()

    test_loader = build_dataloader(config=config,
                                   df=test_df,
                                   transform=transform.valid,
                                   mode='test')

    inf = Inference(config=config,
                    model=model,
                    transform=transform,
                    device=device)
    result = inf.file_inference(data_loader=test_loader)

    test_df['label'] = result
    test_df['label'] = test_df['label'].apply(lambda x: strfy(x))

    test_df.to_csv('sub.csv', index=False)


if __name__ == '__main__':
    main_file('./data/test.csv')
    print("Done !")



