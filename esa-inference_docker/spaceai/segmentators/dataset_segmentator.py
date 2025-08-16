import os


class DatasetSegmentator:
    def __init__(self, root: str):
        super().__init__()
        self.root = root

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)
