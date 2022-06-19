import torch



def apply_ckpt(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt)
    # print(f'모델을 성공적으로 불러왔습니다.')
    return model


def apply_device(model, device):
    import torch.nn as nn

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 10:
            print("Multi-Device")
            torch.nn.DataParallel(model).to(device)
        else:
            model.to(device)
    else:
        model.to(device)
    return model


def efficientnet(model_name="efficientnet-b4", mode="train", num_classes=None):
    from efficientnet_pytorch import EfficientNet
    """
    efficientnet-b0 ~ b7
    """

    assert num_classes is not None, 'config[N_CLASSES] is not exist'

    if mode == "train":
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else:
        override_params = {'num_classes': num_classes}
        model = EfficientNet.from_name(model_name, override_params)

    return model


def unet_load(config, mode="train"):
    # config 로 빼기
    import segmentation_models_pytorch as smp

    if mode == 'train':
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=int(config['MODEL']['IN_CHANNEL']),  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=int(config['MODEL']['N_CLASSES']),      # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )

    else:
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=int(config['MODEL']['IN_CHANNEL']),  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=int(config['MODEL']['N_CLASSES']),  # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )

    return model


# 실제 빌드 함수
def build_model(config, device, mode='train'):
    if config['MODEL']['NAME'] == 'Unet':
        model = unet_load(config, mode)
    elif config['MODEL']['NAME'] == 'Efficient':
        model = efficientnet(mode=mode,
                             num_classes=config['MODEL']['N_CLASSES'])

    if mode == 'train':
        pass
    else:
        print('Load Model... {}'.format(device))
        weight = config['MODEL']['CHECKPOINT']

        if device == 'cpu':
            model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(weight))

    return model.to(device)


# def load_model(config, device, mode,
#                _eval=True, optimizer=None, scheduler=None, keep=False):
#     if config['MODEL']['NAME'] == 'Unet':
#         model = unet_load(device, mode)
#     else:
#         pass
#
#     net = apply_device(model, device)
#
#     saved_dir = config['MODEL']['CHECKPOINT']
#
#     cp = torch.load(saved_dir, map_location=device)
#     if isinstance(net, nn.DataParallel):
#         net.module.load_state_dict(cp["model_state_dict"], strict=False)
#     else:
#         new_dict = remove_prefix(cp["model_state_dict"], "module.")
#         net.load_state_dict(new_dict, strict=False)
#     if _eval:
#         net.eval()
#     if keep:
#         optimizer.load_state_dict(cp["optimizer_state_dict"])
#         if not scheduler:
#             return net, optimizer, None
#         else:
#             scheduler.load_state_dict(cp["scheduler_state_dict"])
#             return net, optimizer, scheduler
#     else:
#         return net, None, None


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


if __name__ == '__main__':
    pass
