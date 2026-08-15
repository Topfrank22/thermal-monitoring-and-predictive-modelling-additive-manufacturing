from config import DATA_DIR
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# DIZIONARIO DEI PROVINI
# Struttura per ogni provino:
#   "seq_name"     : nome del file .seq (senza estensione) e della cartella _frames
#   "stop_frame"   : frame in cui la macchina si ferma (None se STANDARD)
#   "restart_frame": frame in cui la macchina riparte (None se STANDARD)
#   "layers_front" : lista di tuple (start, end) per i layer frontali
#                    None se STANDARD / non ancora annotato
#   "valid"        : False se il provino è da escludere (es. errore di registrazione)
# ---------------------------------------------------------------------------

SPECIMENS = {
    "Rec-022_10s_1": {
        "seq_name": "Rec-022_10s_1",
        "stop_frame": 2440,
        "restart_frame": 2711,
        "layers_front": [(2794, 2806), (2942, 2954), (3075, 3090),(3208, 3223),(3345,3357),(3482,3497)],
        "valid": True,
    },
    "Rec-023": {
        "seq_name": None,
        "stop_frame": None,
        "restart_frame": None,
        "layers_front": None,
        "valid": False,  # ERRORE di registrazione
    },
    "Rec-024_90s_1": {
        "seq_name": "Rec-024_90s_1",
        "stop_frame": 2436,
        "restart_frame": 4721,
        "layers_front": [(4801, 4816), (4943, 4958), (5090, 5101), (5226,5240), (5361,5371),(5497,5511)],
        "valid": True,
    },
    "Rec-025_30s_1": {
        "seq_name": "Rec-025_30s_1",
        "stop_frame": 2315,
        "restart_frame": 3081,
        "layers_front": [(3158, 3170), (3307, 3320), (3445, 3458),(3573,3585), (3708,3722), (3835,3847)],
        "valid": True,
    },
    "Rec-026_60s_2": {
        "seq_name": "Rec-026_60s_2",
        "stop_frame": 2377,
        "restart_frame": 3893,
        "layers_front": [(3974, 3986), (4109, 4124), (4253, 4264),(4386,4400),(4525,4540),(4670,4681)],
        "valid": True,
    },
    "Rec-027_std_2": {
        "seq_name": "Rec-027_std_2",
        "stop_frame": None,
        "restart_frame": None,
        "layers_front": [
            (1125, 1139),   # layer  1
            (1285, 1297),   # layer  2
            (1450, 1463),   # layer  3
            (1612, 1625),   # layer  4
            (1767, 1775),   # layer  5
            (1910, 1921),   # layer  6
            (2058, 2072),   # layer  7
            (2213, 2226),   # layer  8
            (2367, 2377),   # layer  9
            (2514, 2525),   # layer  10
            (2660, 2671),   # layer  11
            (2798, 2813),   # layer  12
            (2936, 2947),   # layer  13
            (3076, 3090),   # layer  14
            (3212, 3224),   # layer  15
            (3354, 3368),   # layer  16
            (3495, 3508),   # layer  17
            (3617, 3632),   # layer  18
            (3745, 3757),   # layer  19
            (3880, 3887),   # layer  20
            (4002, 4017),   # layer  21
            (4132, 4143),   # layer  22
            (4256, 4271),   # layer  23
            (4384, 4397),   # layer  24
            (4506, 4519),   # layer  25
            (4626, 4635),   # layer  26
            (4742, 4751),   # layer  27
            (4855, 4868),   # layer  28
            (4969, 4983),   # layer  29
        ],
        "valid": True,
    },
    "Rec-028_30s_2": {
        "seq_name": "Rec-028_30s_2",
        "stop_frame": 2397,
        "restart_frame": 3170,
        "layers_front": [(3253, 3267), (3397, 3412), (3540, 3552),(3677,3692), (3818,3829), (3956,3969)],
        "valid": True,
    },
    "Rec-029_10s_2": {
        "seq_name": "Rec-029_10s2",
        "stop_frame": 2358,
        "restart_frame": 2618,
        "layers_front": [(2703, 2716), (2847, 2860), (2984, 2995), (3126,3138), (3267,3277), (3399,3409)],
        "valid": True,
    },
    "Rec-030_90s_2": {
        "seq_name": "Rec-030_90s2",
        "stop_frame": 2429,
        "restart_frame": 4693,
        "layers_front": [(4767, 4782), (4911, 4924), (5050, 5062), (5185,5196), (5332,5342), (5465,5478)],
        "valid": True,
    },
    "Rec-031_30s_3": {
        "seq_name": "Rec-031_30s_3",
        "stop_frame": 2456,
        "restart_frame": 3219,
        "layers_front": [(3297, 3310), (3445, 3453), (3585, 3596), (3723,3737),(3866,3880), (4013,4026)],
        "valid": True,
    },
    "Rec-032_30s_4": {
        "seq_name": "Rec-032_30s_4",
        "stop_frame": 2453,
        "restart_frame": 3221,
        "layers_front": [(3294, 3306), (3431, 3445), (3574, 3584), (3715,3727), (3853,3868), (3994,4004)],
        "valid": True,
    },
    "Rec-G3_S60_1": {
        "seq_name": "Rec-G3_S60",
        "stop_frame": 0,
        "restart_frame": 1705,
        "layers_front": [(1802, 1817), (1975, 1990), (2144, 2160), (2311,2326), (2480,2495), (2648,2663)],
        "valid": True,
    },
    "Rec-G3_std_1": {
        "seq_name": "Rec-G3_std",
        "stop_frame": None,
        "restart_frame": None,
        "layers_front": [
            (246,  260),    # layer  1
            (406,  420),    # layer  2
            (571,  586),    # layer  3
            (729,  742),    # layer  4
            (894,  908),    # layer  5
            (1054, 1068),   # layer  6
            (1209, 1222),   # layer  7
            (1367, 1381),   # layer  8
            (1522, 1537),   # layer  9
            (1667, 1682),   # layer  10
            (1820, 1835),   # layer  11
            (1969, 1984),   # layer  12
            (2122, 2137),   # layer  13
            (2267, 2282),   # layer  14
            (2412, 2428),   # layer  15
            (2561, 2576),   # layer  16
            (2700, 2714),   # layer  17
            (2836, 2849),   # layer  18
            (2975, 2990),   # layer  19
            (3110, 3124),   # layer  20
            (3244, 3258),   # layer  21
            (3344, 3359),   # layer  22
        ],
        "valid": True,
    },
}


# ---------------------------------------------------------------------------
# ECCEZIONI CUSTOM
# ---------------------------------------------------------------------------

class SpecimenNotFoundError(KeyError):
    """Il nome del provino non esiste nel dizionario SPECIMENS."""

class SpecimenNotValidError(ValueError):
    """Il provino e' marcato come non valido (es. errore di registrazione)."""

class LayerNotDefinedError(ValueError):
    """I layer del provino non sono ancora stati annotati (es. STANDARD)."""

class LayerIndexError(IndexError):
    """L'indice del layer richiesto non esiste (deve essere tra 1 e N)."""


# ---------------------------------------------------------------------------
# FRAME LOADER
# ---------------------------------------------------------------------------

def get_layer_frames(specimen_name: str, layer_index: int, margin: int = 0) -> dict:
    """
    Restituisce i frame associati a un layer specifico di un provino.

    Parametri
    ----------
    specimen_name : str
        Nome del provino (chiave nel dizionario SPECIMENS).
    layer_index : int
        Indice del layer di interesse: da 1 a N (len(layers_front)).
    margin : int, optional
        Numero di frame da aggiungere prima (pre) e dopo (post) il range core.
        Default = 0 (nessun margine).

    Ritorna
    -------
    dict con le seguenti chiavi:
        - "specimen"    : nome del provino
        - "layer_index" : indice del layer (1..N)
        - "frame_start" : primo frame core (inclusivo)
        - "frame_end"   : ultimo frame core (inclusivo)
        - "frames_pre"  : lista di interi - frame di margine PRIMA del core
        - "frames_core" : lista di interi - frame validi per la misura
        - "frames_post" : lista di interi - frame di margine DOPO il core
        - "frames_dir"  : Path alla cartella contenente le immagini del provino

    Eccezioni
    ---------
    SpecimenNotFoundError  : se specimen_name non e' nel dizionario
    SpecimenNotValidError  : se il provino e' marcato valid=False
    LayerNotDefinedError   : se layers_front e' None (es. STANDARD)
    LayerIndexError        : se layer_index non e' nel range 1..N
    """
    if specimen_name not in SPECIMENS:
        raise SpecimenNotFoundError(f"Provino '{specimen_name}' non trovato nel dizionario.")

    spec = SPECIMENS[specimen_name]

    if not spec["valid"]:
        raise SpecimenNotValidError(
            f"Il provino '{specimen_name}' e' marcato come non valido."
        )

    if spec["layers_front"] is None:
        raise LayerNotDefinedError(
            f"I layer del provino '{specimen_name}' non sono ancora annotati (STANDARD)."
        )

    num_layers = len(spec["layers_front"])
    if layer_index < 1 or layer_index > num_layers:
        raise LayerIndexError(
            f"layer_index deve essere tra 1 e {num_layers}. Ricevuto: {layer_index}"
        )

    start, end = spec["layers_front"][layer_index - 1]

    frames_pre  = list(range(max(0, start - margin), start))
    frames_core = list(range(start, end + 1))
    frames_post = list(range(end + 1, end + 1 + margin))

    frames_dir = Path(DATA_DIR) / f"{spec['seq_name']}_frames"

    return {
        "specimen":    specimen_name,
        "layer_index": layer_index,
        "frame_start": start,
        "frame_end":   end,
        "frames_pre":  frames_pre,
        "frames_core": frames_core,
        "frames_post": frames_post,
        "frames_dir":  frames_dir,
    }


def get_all_layers(specimen_name: str, margin: int = 0) -> list[dict]:
    """
    Restituisce i frame di tutti i layer definiti per un provino.

    Ritorna
    -------
    Lista di N dizionari, uno per layer, nel formato di get_layer_frames().
    """
    if specimen_name not in SPECIMENS:
        raise SpecimenNotFoundError(f"Provino '{specimen_name}' non trovato nel dizionario.")

    spec = SPECIMENS[specimen_name]

    if not spec["valid"]:
        raise SpecimenNotValidError(
            f"Il provino '{specimen_name}' e' marcato come non valido."
        )

    if spec["layers_front"] is None:
        raise LayerNotDefinedError(
            f"I layer del provino '{specimen_name}' non sono ancora annotati (STANDARD)."
        )

    num_layers = len(spec["layers_front"])
    return [get_layer_frames(specimen_name, i, margin) for i in range(1, num_layers + 1)]


def load_frame(frames_dir: Path, frame_index: int) -> np.ndarray:
    """
    Carica un singolo frame da disco come array numpy.

    Il nome file e' il numero del frame zero-padded a 4 cifre (es. 0007.png).

    Parametri
    ----------
    frames_dir  : Path alla cartella del provino
    frame_index : indice del frame da caricare

    Ritorna
    -------
    np.ndarray con i valori del frame.
    """
    from PIL import Image
    frame_path = frames_dir / f"{frame_index:04d}.png"
    return np.array(Image.open(frame_path))
