from mrcnn.config import Config

class OneClassConfig(Config):
    """
    mrcnn.configからConfigクラスを継承して訓練の設定を行うクラス．
    """

    # Config名
    NAME = 'cell_dataset'

    # バッチサイズ
    BATCH_SIZE = 16

    # クラス数 = 背景 + 検出クラス数
    NUM_CLASSES = 1 + 1

    # 1エポック当たりのステップ数
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 5

    # 特定領域のconfidence(:信頼度)が90%以下なら物体検出フェイズをスキップ
    DETECTION_MIN_CONFIDENCE = 0.9
