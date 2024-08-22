from config import Config
from optim.solver import Solver, evaluate_all_heuristics
from tm_utils import (
    ImagePreprocessing,
    NNImageEmbeddings,
    NNSimilarityMeasure,
    ImageEmbeddings,
    SimilarityMeasure,
    ContourImageEmbeddings,
    ContourSimilarityMeasure, Task,
)


class BasicSolver(Solver):
    def solve_one(self):
        selected = self.heuristic.solve()
        return self.env.evaluate_selection_transfer_based(selected)

    def _init_data(self, i):
        super()._init_data(i)
        self.env.update_visual_similarities(self.config.AFFINE_FUNCTIONS)


def run_one_thresh(obj_num, run_num, bgt):
    c = Config()
    c.SUCCESS_RATE_BOOLEAN = True
    c.MH_TIME_BUDGET = 1
    c.MH_BUDGET = bgt
    c.OBJ_NUM = obj_num
    configs = [
        (
            [],
            ImageEmbeddings.DINO_FULL,
            SimilarityMeasure.COSINE,
            False,
            [0.48743718592964824, 0.6331658291457286, 0.7437185929648241],
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
            ContourImageEmbeddings.MASK_RCNN,
            ContourSimilarityMeasure.ASD,
            True,
            [0.04522613065326633, 0.11055276381909548, 0.135678391959799],
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
            NNImageEmbeddings.SIAMESE,
            NNSimilarityMeasure.TRAINED,
            True,
            [0.5376884422110553, 0.49246231155778897, 0.6080402010050251],
        ),
    ]
    for ps, emb, sim, use_all, thresholds in configs:
        c.IMAGE_PREPROCESSING = ps
        c.IMAGE_EMBEDDINGS = emb
        c.SIMILARITY_MEASURE = sim
        c.USE_ALL_IMAGES = use_all
        c.SIMILARITY_THRESHOLDS = thresholds
        results = evaluate_all_heuristics(BasicSolver, c, n=run_num)
        print(f"For config {ps}, {emb}, {sim}, {use_all}")
        for name, mean, std, total_time in results:
            # print(f"{name}: {mean} +/- {std}, time: {total_time}")
            print(f"{name}: {mean}")


def run_one_affine(obj_num, run_num, bgt):
    c = Config()
    c.SUCCESS_RATE_BOOLEAN = False
    c.MH_TIME_BUDGET = 1
    c.MH_BUDGET = bgt
    c.OBJ_NUM = obj_num
    configs = [
        (
            [],
            ImageEmbeddings.DINO_FULL,
            SimilarityMeasure.COSINE,
            False,
            {
                Task.GRASPING: (
                    0.7181944867911111,
                    0.1138801077232438
                ),
                Task.PUSHING: (
                    0.7687713724858166,
                    -0.04494050594772425
                ),
                Task.HAMMERING: (
                    0.5604995163692343,
                    0.09807984839240697
                )
            },
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM],
            ImageEmbeddings.DINO_FULL,
            SimilarityMeasure.COSINE,
            True,
            {
                Task.GRASPING: (
                    0.7227911796537266,
                    0.12563668556229102
                ),
                Task.PUSHING: (
                    0.703188388050649,
                    0.0025273010088077696
                ),
                Task.HAMMERING: (
                    0.5229996181595288,
                    0.11427449671574019
                )
            },
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM, ImagePreprocessing.GREYSCALE],
            NNImageEmbeddings.SIAMESE,
            NNSimilarityMeasure.TRAINED,
            True,
            {
                Task.GRASPING: (
                    0.9498635218251053,
                    0.018646840409686005
                ),
                Task.PUSHING: (
                    0.87386853069105,
                    0.03705817606803352
                ),
                Task.HAMMERING: (
                    0.6954535244565685,
                    0.13072154780451728
                )
            },
        ),
    ]
    for ps, emb, sim, use_all, funs in configs:
        c.IMAGE_PREPROCESSING = ps
        c.IMAGE_EMBEDDINGS = emb
        c.SIMILARITY_MEASURE = sim
        c.USE_ALL_IMAGES = use_all
        c.AFFINE_FUNCTIONS = funs
        results = evaluate_all_heuristics(BasicSolver, c, n=run_num)
        print(f"For config {ps}, {emb}, {sim}, {use_all}")
        for name, mean, std, total_time in results:
            # print(f"{name}: {mean} +/- {std}, time: {total_time}")
            print(f"{name}: {mean}")


if __name__ == "__main__":
    for bgt in range(1000, 10000, 1000):
        print(f"For budget {bgt}")
        run_one_thresh(51, 100, bgt)
        run_one_thresh(40, 100, bgt)

    for bgt in range(1000, 10000, 1000):
        print(f"For budget {bgt}")
        run_one_affine(51, 100, bgt)
        run_one_affine(40, 100, bgt)
