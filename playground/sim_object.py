from playground.extractor import Extractor
from playground.object import Object


class SimObject(Object):
    def __init__(self, index, c, task, name, img_path):
        super().__init__(index, c, task)
        self.name = name
        embeddings = Extractor()(img_path, c)
        if c.USE_ALL_IMAGES:
            self.visible_repr = embeddings
        else:
            # only take the 0th image (top down)
            self.visible_repr = embeddings[0]
        self.image_path = img_path

    def __str__(self):
        return f"{self.name}, {self.visible_repr}, {self.task.value}"
