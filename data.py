import cv2
from PIL import Image
import glob
import os
import copy
import numpy as np

from mrcnn import utils
from mrcnn.model import log

class OneClassDataset(utils.Dataset):
    """
    mrcnn.utilsからDatasetクラスを継承して，
    画像読み込みメソッドの定義とmask生成のメソッドのオーバーライドを行うクラス．
    """

    def load_dataset(self, dataset_dir):
        """
        データセットの登録を行うメソッド．
        """
        # データセット名，クラスID，クラス名
        self.add_class('shrimp_dataset', 1, 'shrimp')

        images = glob.glob(os.path.join(dataset_dir, 'images', '*.png'))
        masks = glob.glob(os.path.join(dataset_dir, 'masks', '*.png'))

        for image_path, mask_path in zip(images, masks):
            assert os.path.basename(image_path) == os.path.basename(mask_path), 'データセット名不一致'

            # 拡張子無しのファイル名(IDとして使う)
            image_path_stem = os.path.basename(image_path)[:-4]

            image = Image.open(image_path)
            height = image.size[0]
            width = image.size[1]

            mask = Image.open(mask_path)

            assert image.size == mask.size, 'サイズ不一致'

            self.add_image(
                'shrimp_dataset',
                path=image_path,
                image_id=image_path_stem,
                mask_path=mask_path,
                width=width,
                height=height
            )
        
    def load_mask(self, image_id):
        """
        maskデータとクラスIDを生成するメソッド．
        """
        image_info = self.image_info[image_id]

        if image_info['source'] != 'shrimp_dataset':
            return super(self.__class__, self).load_mask(image_id)
        
        mask_path = image_info['mask_path']
        mask, class_index = blob_detection(mask_path)

        return mask, class_index

    def image_reference(self, image_id):
        """
        画像のパスを返すメソッド．
        """
        info = self.image_info[image_id]

        if info['source'] == 'shrimp_dataset':
            return info
        else:
            super(self.__class__, self).image_reference(image_id)
    
def blob_detection(mask_path):
    """
    """

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 念のため，もう一度二値化
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    """
    label = [n_labels, label_images, data, center]
    n_labels: ラベル数(0は背景とする)
    label_images: ラベル番号が入った配列データ．
                    座標(x, y)のラベル番号はlabel_images[x, y]で取得可能．
    data: オブジェクトの配列データ．
            x, y, w, h, size = data[ラベル番号]で参照可能．
            x, y, w, h はオブジェクトの外接矩形の左上を基準にしたもの，size は面積(pixel)．
    center: オブジェクトの中心点(float_x, float_y)
    """
    label = cv2.connectedComponentsWithStats(mask)
    data = copy.deepcopy(label[1])  # 全く別のオブジェクトとしてコピー

    labels = []
    for label in np.unique(data):
        # ラベル0は背景
        if label == 0:
            continue
        else:
            labels.append(label)

    mask = np.zeros((mask.shape) + (len(labels),), dtype=np.uint8)

    for n, label in enumerate(labels):
        mask[:, :, n] = np.uint8(data == label)  # ラベルの部分を1に

    class_index = np.ones([mask.shape[-1]], dtype=np.int32)

    return mask, class_index
