import torch
import cv2
import numpy as np
import os
from tqdm import tqdm


class Inference:
    def __init__(self,
                 config=None,
                 model=None,
                 transform=None,
                 device=None,
                 # logger=None
                 ):

        self.config = config
        self.model = model.eval()
        self.transform = transform
        self.device = device
        # self.logger = logger

    def file_inference(self,
                       data_loader=None,
                       ):
        """
        학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
        :param model:
        :param data_loader:
        :param criterion:
        :param optimizer:
        :return: average_loss(float64), total_losses(list), accuracy(float)
        """
        sm = torch.nn.Softmax(dim=1)
        ret_list = []
        model = self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"{'Test :'}")
            for data in progress_bar:
                out = model(data.to(self.device)).float()
                hat = sm(out)

                ans = torch.argmax(hat, dim=1)
                ret_list.extend(ans.tolist())

        return ret_list


    def do_inference(self, img, save=False):
        """
        img : 이미지 경로
        save : True 일때 저장
        """
        with torch.no_grad():
            original = cv2.imread(img, cv2.IMREAD_UNCHANGED)

            h = original.shape[:2][0]
            w = original.shape[:2][1]

            img_value = self.transform.valid(image=original)
            img = img_value['image'].to(self.device).float().unsqueeze(0)

            outputs = self.model(img)
            # print(outputs.shape)
            # print(torch.unique(outputs[0][2]))
            # predicted = torch.argmax(outputs, dim=1)
            # print(predicted.shape)
            # print(torch.unique(predicted))
            outputs = torch.sigmoid(outputs)

            predicted = torch.where(outputs > 0.5, 1, 0)
            predicted = self.transform.restore_size(predicted, (h, w))

        # batch 제거
        np_predicted = predicted[0]

        ori = np.array(original).copy()

        for idx, mask in enumerate(np_predicted):
            ori[mask == 1] = color_list[idx]

        if save:
            os.makedirs('inference_out', exist_ok=True)
            cv2.imwrite('./inference_out/test_img.png', ori)

        #
        # for row in np_predicted:
        #     mask = row.cpu().numpy()
        #     mask.dtype = 'uint8'
        #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        #     masked_img = cv2.fillPoly(original, contours, (255, 0, 0))
        #
        # if save:
        #     os.makedirs('inference_out', exist_ok=True)
        #     cv2.imwrite('./inference_out/test_img.png', masked_img)

        return predicted, ori
