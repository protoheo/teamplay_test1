import numpy as np
import torch
from tqdm import tqdm


def print_result(result):
    """
    결과를 print하는 함수 입니다.
    :param result: list를 input으로 받아 print합니다.
    :return:
    """
    epoch, train_loss, valid_loss, train_acc, valid_acc = result
    print(
        f"[epoch{epoch}] train_loss: {round(train_loss, 3)},"
        f" valid_loss: {round(valid_loss, 3)},"
        f" train_acc: {train_acc},"
        f" valid_acc: {valid_acc}"
    )


def share_loop(epoch=10,
               model=None,
               data_loader=None,
               criterion=None,
               optimizer=None,
               device=None,
               mode="train"):
    """
    학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
    :param epoch:
    :param model:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :param device:
    :param mode: 'train', 'valid' 중 하나의 값을 받아 loop를 진행합니다.
    :return: average_loss(float64), total_losses(list), accuracy(float)
    """

    total_losses = []
    total_acc = []

    if optimizer is not None:
        opt_name = optimizer[1]
        optimizer = optimizer[0]
    mode = mode.lower()
    progress_bar = tqdm(data_loader, desc=f"{mode} {epoch}")
    if mode == "train":
        model.train()
        for batch in progress_bar:
            data, label = batch
            out = model(data.to(device)).float()
            label = label.to(device)

            loss = criterion(out, label)
            # 역전파
            optimizer.zero_grad()
            loss.backward()

            if opt_name == "SAM":
                optimizer.first_step(zero_grad=True)
                criterion(
                    model(data.to(device)).float(), label
                ).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            # accuracy 계산
            prediction = out.argmax(dim=1)
            labels = label.argmax(dim=1)
            acc = torch.mean((prediction == labels).float())

            loss = loss.item()
            acc = acc.item()

            total_losses.append(loss)
            total_acc.append(acc)

            progress_bar.set_postfix({'loss': loss, 'acc': acc})

    elif mode == 'valid':
        model.eval()
        with torch.no_grad():
            for batch in progress_bar:
                data, label = batch
                out = model(data.to(device)).float()
                label = label.to(device)

                loss = criterion(out, label)

                # accuracy 계산
                prediction = out.argmax(dim=1)
                labels = label.argmax(dim=1)
                acc = torch.mean((prediction == labels).float())

                loss = loss.item()
                acc = acc.item()

                total_losses.append(loss)
                total_acc.append(acc)

                progress_bar.set_postfix({'loss': loss, 'acc': acc})
    else:
        raise Exception(f'mode는 train, valid 중 하나여야 합니다. 현재 mode값 -> {mode}')

    avg_loss = np.average(total_losses)
    avg_acc = np.average(total_acc)

    return avg_loss, avg_acc

    # elif mode == 'ensemble':
    #     all_preds = []
    #     for batch in tqdm(data_loader, desc=f"{mode}"):
    #         tmp_out = 0
    #         for model_ in model:
    #             model_.eval()
    #             with torch.no_grad():
    #                 data = batch
    #                 out = model_(data).logits
    #                 soft_out = torch.softmax(out, dim=1)
    #                 tmp_out += soft_out
    #
    #         predicted = torch.argmax(soft_out, dim=1)
    #         all_preds.append(predicted.detach())
    #     return all_preds

