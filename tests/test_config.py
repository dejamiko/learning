from config import Config
from tm_utils import ImagePreprocessing, ImageEmbeddings, ContourImageEmbeddings


def test_get_embedding_spec_default_works():
    config = Config()
    assert config.get_embedding_spec() == "dino_2_full, []"


def test_get_embedding_spec_preprocessing_works():
    config = Config()
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.GREYSCALE]
    assert config.get_embedding_spec() == "dino_2_full, [greyscale]"

    config.IMAGE_PREPROCESSING = [
        ImagePreprocessing.GREYSCALE,
        ImagePreprocessing.SEGMENTATION,
        ImagePreprocessing.BACKGROUND_REM,
    ]
    assert (
        config.get_embedding_spec()
        == "dino_2_full, [greyscale, segmentation, background_rem]"
    )


def test_get_embedding_spec_different_image_emb_works():
    config = Config()
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DOBBE
    assert config.get_embedding_spec() == "dobbe, []"


def test_get_embedding_spec_contour_image_emb_works():
    config = Config()
    config.IMAGE_EMBEDDINGS = ContourImageEmbeddings.CASCADE_MASK_RCNN
    assert config.get_embedding_spec() == "cascade_mask_rcnn, []"
