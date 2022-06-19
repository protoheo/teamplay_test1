import cv2
import json
import numpy as np


# mask 이미지 생성 함수
def make_mask(json_path):
    with open(json_path, 'r') as r:
        image_info = json.load(r)

    lst_coordinate = []
    # polygon 좌표 인식
    for coordinate in image_info['annotations'][0]['label']['data']:
        lst_coordinate.append([coordinate['x'], coordinate['y']])

    img_path = json_path.replace('json', 'jpg')

    image_array = cv2.imread(img_path)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # green_color = (0, 225, 0)
    # draw_img = image_array.copy()

    polygon_xy = np.array(lst_coordinate, np.int32)
    # draw_img = cv2.fillPoly(draw_img, [polygon_xy], green_color)

    # 클래스 별 mask 생성
    if 'garibi\\' in img_path:
        cls_num = 1
    else:
        cls_num = 2

    zero_mask = np.zeros(image_array.shape[0:2])
    masked_polygon = cv2.fillPoly(zero_mask, [polygon_xy], cls_num)

    return masked_polygon, img_path
