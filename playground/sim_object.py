from playground.extractor import Extractor
from playground.object import Object


class SimObject(Object):
    def __init__(self, index, c, task, name, img_path):
        super().__init__(index, c, task)
        self.name = name
        self.visible_repr = Extractor()(img_path, c)[0]

    def __str__(self):
        return f"{self.name}, {self.visible_repr}, {self.task.value}"
