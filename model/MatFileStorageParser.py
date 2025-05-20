from scipy.io import savemat, loadmat
import numpy as np
from numpy.typing import NDArray

class FileModel:
    BaseStations: list[tuple[float, float, float]] # (x,y,ptx)
    NFemtoCells: int
    NMacroCells: int
    
    @staticmethod
    def save_file_basic(file_path: str, BaseStations: list[tuple[float, float, float]], NFemtoCells: int, NMacroCells: int):
        complete_data = {
            "BaseStations": BaseStations,
            "NFemtoCells": NFemtoCells,
            "NMacroCells": NMacroCells
        }
        savemat(file_path, complete_data)

    @staticmethod
    def load_file_basic(file_path: str):
        data = loadmat(file_path)
        return data["BaseStations"], data["NFemtoCells"][0][0], data["NMacroCells"][0][0]

    @staticmethod
    def save_file_complete(
            file: str,
            BaseStations: list[tuple[float, float, float]],
            NFemtoCells: int,
            NMacroCells: int,
            NodeAdjacencyMatrix: NDArray[np.float64],
            NodeDijsktraDistanceMatrix: NDArray[np.float64],
            NodePositions: NDArray[np.float64],
            NodeTypeHL4: NDArray[np.int_],
            NodeTypeHL5: NDArray[np.int_],
            TrafficInjection: NDArray[np.float64],
            RangeForFemtoCells: NDArray[np.float64],
            PowerForFemtoCells: NDArray[np.float64],
            PowerForMacrocells: NDArray[np.float64],
            PowerForHPLD: NDArray[np.float64],
            AlphaLoss: float,
            EuclideanToKmScale: float,
            NodesWithHPLD: NDArray[np.int_],
            NodesWithFemtoCells: NDArray[np.int_],
            StatTotalHPLD: int,
            StatToalFemtoCells: int,
            StatDeployedBwOfFemtoCells: float,
            StatEstimatedThrougputOfFemtoCellsTrafficProfileBased: float,
            StatTotalCoverageAreaOfFemtoCells: float,
            StatTotalCoverageAreaPercentageOfFemtoCells: float,
        ):
        # Parse from NDArray to list when needed
        data = {
            "BaseStations": BaseStations,
            "NFemtoCells": NFemtoCells,
            "NMacroCells": NMacroCells,
            "NodeAdjacencyMatrix": NodeAdjacencyMatrix.tolist(),
            "NodeDijsktraDistanceMatrix": NodeDijsktraDistanceMatrix.tolist(),
            "NodePositions": NodePositions.tolist(),
            "NodeTypeHL4": NodeTypeHL4.tolist(),
            "NodeTypeHL5": NodeTypeHL5.tolist(),
            "TrafficInjection": TrafficInjection.tolist(),
            "RangeForFemtoCells": RangeForFemtoCells.tolist(),
            "PowerForFemtoCells": PowerForFemtoCells.tolist(),
            "PowerForMacrocells": PowerForMacrocells.tolist(),
            "PowerForHPLD": PowerForHPLD.tolist(),
            "AlphaLoss": AlphaLoss,
            "EuclideanToKmScale": EuclideanToKmScale,
            "NodesWithHPLD": NodesWithHPLD.tolist(),
            "NodesWithFemtoCells": NodesWithFemtoCells.tolist(),
            "StatTotalHPLD": StatTotalHPLD,
            "StatToalFemtoCells": StatToalFemtoCells,
            "StatDeployedBwOfFemtoCells": StatDeployedBwOfFemtoCells,
            "StatEstimatedThrougputOfFemtoCellsTrafficProfileBased": StatEstimatedThrougputOfFemtoCellsTrafficProfileBased,
            "StatTotalCoverageAreaOfFemtoCells": StatTotalCoverageAreaOfFemtoCells,
            "StatTotalCoverageAreaPercentageOfFemtoCells": StatTotalCoverageAreaPercentageOfFemtoCells,
        }
        savemat(file, data)
