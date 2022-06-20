import numpy as np
from scipy import stats
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F


class ImageTransform:
    def __init__(self):
        self.train = A.Compose([
                A.Resize(380, 380, always_apply=True),
                # A.RandomCrop(height=200, width=200, p=0.2),
                # A.PadIfNeeded(min_height=400, min_width=400, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                A.VerticalFlip(p=0.2),
                # A.Blur(p=0.2),
                # A.RandomRotate90(p=0.2),
                # A.ShiftScaleRotate(p=0.2, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(p=0.2),
                # A.RandomSunFlare(p=0.2, src_radius=200),
                # A.RandomShadow(p=0.2),
                # A.RandomFog(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                # transpose_mask=True
            ]
        )
        self.valid = A.Compose(
            [
                A.Resize(380, 380, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )

    def restore_size(self, mask, shape):
        return F.resize(mask, shape)

    def cutmix(self, images, labels):
        lmbda = stats.beta(1, 1).rvs(1)[0]
        H = images[0].size()[-2]
        W = images[0].size()[-1]
        cut_rat = np.sqrt(1 - lmbda)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        center_x = np.random.randint(W)
        center_y = np.random.randint(H)
        boundary_x1 = np.clip(center_x - cut_w // 2, 0, W)
        boundary_x2 = np.clip(center_x + cut_w // 2, 0, W)
        boundary_y1 = np.clip(center_y - cut_h // 2, 0, H)
        boundary_y2 = np.clip(center_y + cut_h // 2, 0, H)

        adjusted_lmbda = 1 - (
            (boundary_x2 - boundary_x1) * (boundary_y2 - boundary_y1)
        ) / (images.size(-1) * images.size(-2))

        random_idx = np.random.permutation(images.size(0))
        shuffled_labels = labels[random_idx]
        new_patches = images[
            random_idx, :, boundary_y1:boundary_y2, boundary_x1:boundary_x2
        ]
        images[:, :, boundary_y1:boundary_y2, boundary_x1:boundary_x2] = new_patches
        return images, shuffled_labels, adjusted_lmbda


if __name__ == "__main__":
    pass
