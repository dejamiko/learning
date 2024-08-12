import json
import os

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from dino_vit_features.extractor import ViTExtractor
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamPredictor
from timm.data.transforms_factory import create_transform
from transformers import (
    AutoImageProcessor,
    AutoModel,
    ViTImageProcessor,
    ViTModel,
    SwinModel,
    ViTMSNModel,
)

from tm_utils import (
    ImageEmbeddings,
    ContourImageEmbeddings,
    ImagePreprocessing,
    NNSimilarityMeasure,
)
from vc_models.models.vit import model_utils


class Extractor:
    def __call__(self, img_path, config):
        try:
            return self._load_embeddings(img_path, config)
        except FileNotFoundError:
            num_found = 0
        except KeyError as e:
            num_found = int(str(e)[-2])
        images = self._load_images(img_path, config, num_found)
        self._extract_and_save_embeddings(images, img_path, config, num_found)
        return self._load_embeddings(img_path, config)

    @staticmethod
    def _load_images(img_dir, config, num_found=0):
        assert os.path.exists(
            img_dir
        ), f"The provided image path {img_dir} does not exist"
        assert os.path.isdir(
            img_dir
        ), f"The provided image path {img_dir} is not a directory"

        images_with_names = []
        index = 0
        for filename in sorted(os.listdir(img_dir)):
            if not filename.endswith(".png"):
                continue
            if index < num_found:
                index += 1
                continue
            image = cv2.imread(os.path.join(img_dir, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Extractor._apply_preprocessing(
                config, image, os.path.join(img_dir, filename)
            )
            images_with_names.append((image, filename))
        return images_with_names

    @staticmethod
    def _apply_preprocessing(config, image, filename):
        for pre in config.IMAGE_PREPROCESSING:
            match pre:
                case ImagePreprocessing.BACKGROUND_REM:
                    image = Extractor._preprocess_background_rem(image)
                case ImagePreprocessing.SEGMENTATION:
                    image = Extractor._preprocess_segmentation(image, config)
                case ImagePreprocessing.GREYSCALE:
                    image = Extractor._preprocess_greyscale(image)
                case ImagePreprocessing.CROPPING:
                    image = Extractor._preprocess_cropping(image, filename)
        return image

    def _extract_and_save_embeddings(
        self, images_with_names, img_dir, config, num_found=0
    ):
        if len(images_with_names) == 0:
            raise ValueError(f"The directory provided {img_dir} contains no images")
        for i, (image, n) in enumerate(images_with_names):
            self._save_embedding(
                img_dir,
                self._extract_embeddings(image, os.path.join(img_dir, n), config),
                num_found + i,
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
                return self._extract_dino_2_full(image, config)
            case ImageEmbeddings.VIT:
                return self._extract_vit(image, config)
            case ImageEmbeddings.CONVNET:
                return self._extract_convnet(image, config)
            case ImageEmbeddings.SWIN:
                return self._extract_swin(image, config)
            case ImageEmbeddings.VIT_MSN:
                return self._extract_vit_msn(image, config)
            # robotics specific models
            case ImageEmbeddings.DOBBE:
                return self._extract_dobbe(image, config)
            case ImageEmbeddings.VC:
                return self._extract_vc(image, config)
            # contour models
            case ContourImageEmbeddings.MASK_RCNN:  # pragma: no cover
                return self._extract_mask_rcnn(image, config.MASK_RCNN_THRESHOLD)
            case ContourImageEmbeddings.PANOPTIC_FPN:  # pragma: no cover
                return self._extract_panoptic_fpn(image, config.PANOPTIC_FPN_THRESHOLD)
            case ContourImageEmbeddings.CASCADE_MASK_RCNN:  # pragma: no cover
                return self._extract_cascade_mask_rcnn(
                    image, config.CASCADE_MASK_RCNN_THRESHOLD
                )
            # own trained models
            case ImageEmbeddings.SIAMESE:
                return self._extract_siamese(img_path, config)
        raise ValueError(f"The method provided {config.IMAGE_EMBEDDINGS} is unknown.")

    @staticmethod
    def _save_embedding(img_path, embedding, index, config):
        if os.path.exists(
            os.path.join(img_path, f"embeddings_{config.get_embedding_spec()}.json")
        ):
            with open(
                os.path.join(
                    img_path, f"embeddings_{config.get_embedding_spec()}.json"
                ),
                "r",
            ) as f:
                current_embeddings = json.load(f)
            current_embeddings.append(embedding)
            with open(
                os.path.join(
                    img_path, f"embeddings_{config.get_embedding_spec()}.json"
                ),
                "w",
            ) as f:
                json.dump(current_embeddings, f)
        else:
            with open(
                os.path.join(
                    img_path, f"embeddings_{config.get_embedding_spec()}.json"
                ),
                "w",
            ) as f:
                if index != 0:  # pragma: no cover
                    raise ValueError(
                        "The index cannot be different than 0 if there is no embeddings file"
                    )
                json.dump([embedding], f)

    @staticmethod
    def _load_embeddings(img_path, config):
        assert os.path.exists(img_path), f"The provided path {img_path} does not exist"
        assert os.path.isdir(
            img_path
        ), f"The provided path {img_path} is not a directory"
        with open(
            os.path.join(
                img_path,
                f"embeddings_{config.get_embedding_spec()}.json",
            ),
            "r",
        ) as f:  # this can raise FileNotFound
            all_image_embeddings = json.load(f)
        if len(all_image_embeddings) < 5:
            raise KeyError(
                f"Not all embeddings are present, found {len(all_image_embeddings)}"
            )
        if config.IMAGE_EMBEDDINGS in ContourImageEmbeddings:
            return all_image_embeddings

        try:
            return np.array(all_image_embeddings)
        except ValueError as e:
            assert str(e).startswith(
                "setting an array element with a sequence. "
                "The requested array has an inhomogeneous shape"
            )
            return all_image_embeddings

    @staticmethod
    def _extract_dino(image_path, layer, config):
        extractor = ViTExtractor(config.MODEL_TYPE, config.STRIDE, device=config.DEVICE)
        image_batch, _ = extractor.preprocess(image_path, config.LOAD_SIZE)
        embedding = (
            extractor.extract_descriptors(
                image_batch.to(config.DEVICE), layer, config.FACET, config.BIN
            )
            .detach()
            .cpu()
            .numpy()
        )
        embedding = embedding.squeeze().mean(axis=0)
        return embedding.tolist()

    @staticmethod
    def _extract_dino_full(image, config):
        dino = torch.hub.load("facebookresearch/dino:main", config.MODEL_TYPE).to(
            config.DEVICE
        )

        image_transforms = T.Compose(
            [
                T.Resize(config.LOAD_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        image = Image.fromarray(image)
        img = image_transforms(image)
        img = img.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            img_emb = dino(img).squeeze()
        return img_emb.detach().cpu().numpy().tolist()

    @staticmethod
    def _extract_dino_2_full(image, config):
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(config.DEVICE)

        inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        cls_token = outputs.pooler_output
        return cls_token.flatten().detach().cpu().numpy().tolist()

    @staticmethod
    def _extract_vit(image, config):
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(
            config.DEVICE
        )
        inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0)
        return last_hidden_states.detach().cpu().numpy().tolist()

    @staticmethod
    def _extract_convnet(image, config):
        model = timm.create_model(
            "convnextv2_base.fcmae",
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to(config.DEVICE)
        model.eval()

        image = Image.fromarray(image)

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        output = model.forward_features(
            transforms(image).unsqueeze(0).to(config.DEVICE)
        )
        return (
            model.forward_head(output, pre_logits=True)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    @staticmethod
    def _extract_swin(image, config):
        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-small-patch4-window7-224"
        )
        model = SwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224").to(
            config.DEVICE
        )

        inputs = image_processor(image, return_tensors="pt").to(config.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.detach().squeeze().cpu().numpy().mean(axis=0).tolist()

    @staticmethod
    def _extract_vit_msn(image, config):
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        model = ViTMSNModel.from_pretrained("facebook/vit-msn-small").to(config.DEVICE)

        inputs = image_processor(images=image, return_tensors="pt").to(config.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        return (
            outputs.last_hidden_state.detach().cpu().numpy().squeeze().mean(0).tolist()
        )

    @staticmethod
    def _extract_dobbe(image, config):
        model = timm.create_model("hf_hub:notmahi/dobb-e", pretrained=True).to(
            config.DEVICE
        )

        model.eval()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        image = Image.fromarray(image)

        outputs = model(transforms(image).unsqueeze(0).to(config.DEVICE))

        outputs = outputs.detach().cpu().numpy().squeeze()
        return outputs.tolist()

    @staticmethod
    def _extract_vc(image, config):
        model, embd_size, model_transforms, model_info = model_utils.load_model(
            model_utils.VC1_BASE_NAME
        )

        model.to(config.DEVICE)

        image = Image.fromarray(image)

        transformed_img = model_transforms(image)
        return (
            model(transformed_img.unsqueeze(0).to(config.DEVICE))
            .detach()
            .cpu()
            .squeeze()
            .numpy()
            .tolist()
        )

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

        # NOTE This only runs on gpu
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

        if len(contours) == 0:
            return []

        return np.vstack(contours).squeeze().tolist()

    @staticmethod
    def _preprocess_background_rem(image):
        def bincount_app(a):
            a2D = a.reshape(-1, a.shape[-1])
            col_range = (256, 256, 256)
            a1D = np.ravel_multi_index(a2D.T, col_range)
            return np.unravel_index(np.bincount(a1D).argmax(), col_range)

        dominant_colour = np.array(bincount_app(image))

        lower_bound = dominant_colour - (1, 1, 1)
        upper_bound = dominant_colour + (1, 1, 1)

        mask = cv2.inRange(image, lower_bound, upper_bound)

        masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

        return masked_image

    @staticmethod
    def _preprocess_segmentation(image, config):
        chkpt_path = hf_hub_download(
            "ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth"
        )
        sam = sam_model_registry["vit_b"](checkpoint=chkpt_path)
        sam.to(device=config.DEVICE)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        # Define a region of interest (ROI)
        height, width, _ = image.shape

        # Generate points within the ROI
        num_points = 10
        x_points = np.random.randint(0, height, size=num_points)
        y_points = np.random.randint(0, width, size=num_points)
        point_coords = np.stack((x_points, y_points), axis=1)

        # Predict the mask for the object(s) in the ROI
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=np.ones(num_points)
        )

        # Combine all masks into one (logical OR)
        combined_mask = np.any(masks, axis=0)

        inverted_mask = np.logical_not(combined_mask).astype(np.uint8)

        final_mask = cv2.morphologyEx(
            inverted_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )

        # Create an empty image and apply the combined mask
        isolated_image = np.zeros_like(image)
        isolated_image[final_mask > 0] = image[final_mask > 0]
        return isolated_image

    @staticmethod
    def _preprocess_greyscale(image):
        # this has to be done twice to have a grey but 3-channel image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    @staticmethod
    def _preprocess_cropping(image, filename):
        if filename[0] == "/":
            filename = filename[1:]
        obj_and_task, img_name = filename.split("/")[-2:]
        img_num = int(img_name[6])  # image_[num].png

        img_size = image.shape[0]  # assume square image

        def crop_mid(image):
            return image[
                int(img_size * 0.1) : int(img_size * 0.9),
                int(img_size * 0.1) : int(img_size * 0.9),
            ]

        def crop_top(image):
            return image[
                0 : int(img_size * 0.8), int(img_size * 0.1) : int(img_size * 0.9)
            ]

        def crop_bottom(image):
            return image[
                int(img_size * 0.2) : img_size,
                int(img_size * 0.1) : int(img_size * 0.9),
            ]

        def crop_left(image):
            return image[
                int(img_size * 0.1) : int(img_size * 0.9), 0 : int(img_size * 0.8)
            ]

        def crop_right(image):
            return image[
                int(img_size * 0.1) : int(img_size * 0.9),
                int(img_size * 0.2) : img_size,
            ]

        match img_num:
            case 1:
                return crop_top(image)
            case 2:
                return crop_bottom(image)
            case 3:
                return crop_left(image)
            case 4:
                return crop_right(image)

        if obj_and_task.find("pushing") != -1:
            return crop_top(image)
        return crop_mid(image)

    @staticmethod
    def _extract_siamese(img_path, config):
        all_sims = {}
        for sm in NNSimilarityMeasure:
            sims_sm = {}
            with open(
                os.path.join(
                    "_data/siamese_similarities",
                    f"similarities_{sm.value}_{config.IMAGE_PREPROCESSING}.json",
                ),
                "r",
            ) as f:
                data = json.load(f)

            for k, v in data.items():
                if img_path.startswith(k.split(",")[0]):
                    sims_sm[k.split(",")[1][:-1]] = v
            all_sims[sm.value] = sims_sm

        return all_sims
