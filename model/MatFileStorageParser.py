from scipy.io import savemat, loadmat


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
    def save_file_complete(file: str):
        pass # TODO: all the variables result of the dimensioning algorithm
    
    