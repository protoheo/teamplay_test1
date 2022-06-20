import torch
from tqdm import tqdm


def test_loop(model=None,
              data_loader=None,
              device=None
              ):
    """
    학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
    :param model:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :return: average_loss(float64), total_losses(list), accuracy(float)
    """

    total_acc = []
    total_pred = []
    total_label = []
    progress_bar = tqdm(data_loader, desc=f"{'Test :'}")
    model.eval()
    with torch.no_grad():
        for batch in progress_bar:
            data, label = batch
            out = model(data.to(device)).float()
            label = label.to(device)

            prediction = out.argmax(dim=1)
            labels = label.argmax(dim=1)

            acc = torch.mean((prediction == labels).float()).item()

            total_pred.extend(prediction.tolist())
            total_label.extend(labels.tolist())
            total_acc.append(acc)

    return total_acc, total_pred, total_label

