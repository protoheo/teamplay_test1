import yaml
import os


def load_config(cfg_path):
    return yaml.full_load(open(cfg_path, 'r', encoding='utf-8-sig'))


# 폴더 파일리스트 READ 함수
def find_all_files(root_dir, ret_list=[]):
    for it in os.scandir(root_dir):
        if it.is_file():
            ret_list.append(it.path)

        if it.is_dir():
            ret_list = find_all_files(it, ret_list)
    return ret_list


# 파일 타입 확인 함수
def get_type(path_list, type_list=[]):
    ret_list = []
    if len(type_list) > 0:
        for path_row in path_list:
            ext_split = path_row.split(".")[-1]

            if ext_split in type_list:
                ret_list.append(path_row)
    else:
        ret_list = path_list

    return ret_list


# dataset split
def split_dataset(data_path,
                  target_column=None,
                  valid_size=0.2,
                  seed=None):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """
    학습 데이터셋과 검증 데이터셋으로 나누는 함수입니다.
    :param data_path:
    :param test_size:
    :param seed:
    :param target_column:
    :return:
    """
    df = pd.read_csv(data_path)
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=df[target_column]
    )

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    return train_df, valid_df


def split_datasetv2(data_path,
                    target_column=None,
                    save=False,
                    seed=None):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv(data_path)
    train_df, tmp = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
        stratify=df[target_column]
    )

    valid_df, test_df = train_test_split(
        tmp,
        test_size=0.5,
        random_state=seed,
        stratify=tmp[target_column]
    )

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if save:
        train_df.to_csv('data/split/split_train.csv', index=False)
        valid_df.to_csv('data/split/split_valid.csv', index=False)
        test_df.to_csv('data/split/split_test.csv', index=False)

    return train_df, valid_df, test_df

