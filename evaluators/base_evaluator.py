from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, sentences: list[str], matched_filenames: list[list[str]]) -> None:
        """
        Takes test-sentences and their matched image filenames (k per sentence) and writes result to output-folder.
        """
        pass
