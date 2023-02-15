from abc import abstractmethod, ABC

class DataAnalysis(ABC):
    """
    Abstract class to implement 
    """

    @abstractmethod
    def analyze(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_result(self) -> None:
        pass
