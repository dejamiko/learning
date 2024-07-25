from object import Object


class SimObject(Object):
    def __init__(self, c, name, task):
        super().__init__(c, task)
        self.name = name
        self._load_images()

    def get_visual_similarity(self, other):
        pass

    def __str__(self):
        pass

    def _load_images(self):
        # load the related images for the object with name=`self.name` and task=`self.task` and calculate their DINO
        # embeddings
        # those will be used for the visual similarity calculation
        pass
