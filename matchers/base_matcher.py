from abc import ABC, abstractmethod

class BaseMatcher(ABC):
    @abstractmethod
    def match(self, sentences: list[str], tok_k: int = 1) -> list[list[str]]:
        """
        Takes a list of sentences and returns a list of list of matched image filenames.
        """
        pass
