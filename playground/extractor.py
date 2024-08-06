import json
import os

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from dino_vit_features.extractor import ViTExtractor
from timm.data.transforms_factory import create_transform
from transformers import (
    AutoImageProcessor,
    AutoModel,
    ViTImageProcessor,
    ViTModel,
    SwinModel,
    ViTMSNModel,
)

from vc_models.models.vit import model_utils

from tm_utils import ImageEmbeddings, ContourImageEmbeddings


class Extractor:
    def __call__(self, img_path, config):
        try:
            return self._load_embeddings(img_path, config)
        except (FileNotFoundError, KeyError):
            images = self._load_images(img_path)
            self._extract_and_save_embeddings(images, img_path, config)
            return self._load_embeddings(img_path, config)

    @staticmethod
    def _load_images(img_dir):
        assert os.path.exists(
            img_dir
        ), f"The provided image path {img_dir} does not exist"
        assert os.path.isdir(
            img_dir
        ), f"The provided image path {img_dir} is not a directory"

        images_with_names = []
        for im in sorted(os.listdir(img_dir)):
            if not im.endswith(".png"):
                continue
            images_with_names.append((cv2.imread(os.path.join(img_dir, im)), im))
        return images_with_names

    def _extract_and_save_embeddings(self, images_with_names, img_dir, config):
        if len(images_with_names) == 0:
            raise ValueError(f"The directory provided {img_dir} contains no images")
        for i, (image, n) in enumerate(images_with_names):
            self._save_embedding(
                img_dir,
                self._extract_embeddings(image, os.path.join(img_dir, n), config),
                i,
                config,
            )

    def _extract_embeddings(self, image, img_path, config):
        match config.IMAGE_EMBEDDINGS:
            case ImageEmbeddings.DINO_LAYER_9:
                return self._extract_dino(img_path, 9, config)
            case ImageEmbeddings.DINO_LAYER_11:
                return self._extract_dino(img_path, 11, config)
            case ImageEmbeddings.DINO_FULL:
                return self._extract_dino_full(image, config)
            case ImageEmbeddings.DINO_2_FULL:
                return self._extract_dino_2_full(image)
            case ImageEmbeddings.VIT:
                return self._extract_vit(image)
            case ImageEmbeddings.CONVNET:
                return self._extract_convnet(image)
            case ImageEmbeddings.SWIN:
                return self._extract_swin(image)
            case ImageEmbeddings.VIT_MSN:
                return self._extract_vit_msn(image)
            # robotics specific models
            case ImageEmbeddings.DOBBE:
                return self._extract_dobbe(image)
            case ImageEmbeddings.VC:
                return self._extract_vc(image)
            # contour models
            case ContourImageEmbeddings.MASK_RCNN:  # pragma: no cover
                return self._extract_mask_rcnn(image, config.MASK_RCNN_THRESHOLD)
            case ContourImageEmbeddings.PANOPTIC_FPN:  # pragma: no cover
                return self._extract_panoptic_fpn(image, config.PANOPTIC_FPN_THRESHOLD)
            case ContourImageEmbeddings.CASCADE_MASK_RCNN:  # pragma: no cover
                return self._extract_cascade_mask_rcnn(
                    image, config.CASCADE_MASK_RCNN_THRESHOLD
                )
        raise ValueError(f"The method provided {config.IMAGE_EMBEDDINGS} is unknown.")

    @staticmethod
    def _save_embedding(img_path, embedding, index, config):
        if os.path.exists(os.path.join(img_path, "embeddings.json")):
            with open(os.path.join(img_path, "embeddings.json"), "r") as f:
                current_embeddings = json.load(f)
            if index < len(current_embeddings):
                current_embeddings[index][config.IMAGE_EMBEDDINGS.value] = embedding
            else:
                current_embeddings.append({config.IMAGE_EMBEDDINGS.value: embedding})
            with open(os.path.join(img_path, "embeddings.json"), "w") as f:
                json.dump(current_embeddings, f)
        else:
            with open(os.path.join(img_path, "embeddings.json"), "w") as f:
                if index != 0:  # pragma: no cover
                    raise ValueError(
                        "The index cannot be different than 0 if there is no embeddings file"
                    )
                json.dump([{config.IMAGE_EMBEDDINGS.value: embedding}], f)

    @staticmethod
    def _load_embeddings(img_path, config):
        assert os.path.exists(img_path), f"The provided path {img_path} does not exist"
        assert os.path.isdir(
            img_path
        ), f"The provided path {img_path} is not a directory"
        embeddings = []
        with open(
            os.path.join(
                img_path,
                "embeddings.json",
            ),
            "r",
        ) as f:  # this can raise FileNotFound
            all_image_embeddings = json.load(f)
        for image_embeddings in all_image_embeddings:
            embeddings.append(
                image_embeddings[config.IMAGE_EMBEDDINGS.value]
            )  # this can raise KeyError?
        return np.array(embeddings)

    @staticmethod
    def _extract_dino(image_path, layer, config):
        extractor = ViTExtractor(config.MODEL_TYPE, config.STRIDE, device=config.DEVICE)
        image_batch, _ = extractor.preprocess(image_path, config.LOAD_SIZE)
        embedding = (
            extractor.extract_descriptors(
                image_batch.to(config.DEVICE), layer, config.FACET, config.BIN
            )
            .detach()
            .numpy()
        )
        embedding = embedding.squeeze().mean(axis=0)
        return embedding.tolist()

    @staticmethod
    def _extract_dino_full(image, config):
        dino = torch.hub.load("facebookresearch/dino:main", config.MODEL_TYPE)

        image_transforms = T.Compose(
            [
                T.Resize(config.LOAD_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        image = Image.fromarray(image)
        img = image_transforms(image)
        img = img.unsqueeze(0)
        with torch.no_grad():
            img_emb = dino(img).squeeze()
        return img_emb.tolist()

    @staticmethod
    def _extract_dino_2_full(image):
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        cls_token = outputs.pooler_output
        return cls_token.flatten().tolist()

    @staticmethod
    def _extract_vit(image):
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0)
        return last_hidden_states.detach().numpy().tolist()

    @staticmethod
    def _extract_convnet(image):
        model = timm.create_model(
            "convnextv2_base.fcmae",
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = model.eval()

        image = Image.fromarray(image)

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        output = model.forward_features(transforms(image).unsqueeze(0))
        return (
            model.forward_head(output, pre_logits=True)
            .squeeze()
            .detach()
            .numpy()
            .tolist()
        )

    @staticmethod
    def _extract_swin(image):
        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-small-patch4-window7-224"
        )
        model = SwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224")

        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.detach().squeeze().numpy().mean(axis=0).tolist()

    @staticmethod
    def _extract_vit_msn(image):
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state.detach().numpy().squeeze().mean(0).tolist()

    @staticmethod
    def _extract_dobbe(image):
        model = timm.create_model("hf_hub:notmahi/dobb-e", pretrained=True)

        model.eval()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        image = Image.fromarray(image)

        outputs = model(transforms(image).unsqueeze(0))

        outputs = outputs.detach().numpy().squeeze()
        return outputs.tolist()

    @staticmethod
    def _extract_vc(image):
        model, embd_size, model_transforms, model_info = model_utils.load_model(
            model_utils.VC1_BASE_NAME
        )

        image = Image.fromarray(image)

        transformed_img = model_transforms(image)
        return model(transformed_img.unsqueeze(0)).detach().squeeze().numpy().tolist()

    def _extract_mask_rcnn(self, image, threshold):  # pragma: no cover
        return self._extract_detectron(
            image, "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", threshold
        )

    def _extract_panoptic_fpn(self, image, threshold):  # pragma: no cover
        return self._extract_detectron(
            image, "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", threshold
        )

    def _extract_cascade_mask_rcnn(self, image, threshold):  # pragma: no cover
        return self._extract_detectron(
            image, "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", threshold
        )

    @staticmethod
    def _extract_detectron(image, model, threshold):  # pragma: no cover
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.model_zoo import model_zoo
        # This only runs on gpu
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        predictor = DefaultPredictor(cfg)

        # Perform inference
        outputs = predictor(image)
        # Get the predicted masks
        pred_masks = outputs["instances"].pred_masks.to("cpu").numpy()

        # Create a blank mask with the same dimensions as the input image
        contour_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Get the contours from the binary masks and draw them on the blank mask
        for mask in pred_masks:
            # Convert the mask to uint8 type
            mask = (mask * 255).astype(np.uint8)
            # Find contours
            contours_info, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours_info:
                # Draw the contours on the blank mask
                cv2.drawContours(
                    contour_image, [contour], -1, (255), 2
                )  # Draw in white

        _, binary = cv2.threshold(contour_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return np.vstack(contours).squeeze().tolist()
