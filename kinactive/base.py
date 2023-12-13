from pathlib import Path

from toolz import valmap


SEQ_MODEL_NAMES = {
    "TkST": "TkST_XGB_classifier",
    "AAIO_Tk": "AAIO_Tk_XGB_classifier",
    "AAIO_STk": "AAIO_STk_XGB_classifier",
    "AHAO_Tk": "AHAO_Tk_XGB_classifier",
    "AHAO_STk": "AHAO_Tk_XGB_classifier",
}
STR_MODEL_NAMES = {"kinactive": "kinactive_classifier", "DFG": "DFG_classifier"}

RESOURCES_PATH = Path(__file__).parent / "resources"
MODELS_PATHS = RESOURCES_PATH / "models"
SEQ_MODEL_PATHS = valmap(lambda x: MODELS_PATHS / x, SEQ_MODEL_NAMES)
STR_MODEL_PATHS = valmap(lambda x: MODELS_PATHS / x, STR_MODEL_NAMES)
SEQ_CLASSES = {0: "STK", 1: "TK"}
DFG_STR_CLASSES = {0: "DFG-in", 1: "DFG-other", 2: "DFG-out"}
DFG_SEQ_CLASSES = {0: "DFG-in", 1: "DFG-out"}

TK_PROFILE_PATH = RESOURCES_PATH / "PF07714.hmm"
PK_PROFILE_PATH = RESOURCES_PATH / "PF00069.hmm"

DATA_LINKS_PATH = RESOURCES_PATH / "data_links.json"


if __name__ == "__main__":
    raise RuntimeError
