"""
run_all_plots.py
-----------------
Runner multi-dataset per tutti gli script di analisi.

Configurazione centralizzata: definisci una lista di dataset da processare
e le impostazioni per ogni tipo di grafico. Il runner cicla su tutti i
dataset e genera i grafici nella rispettiva cartella di output:
    Analisi_dataset/<DATASET_NAME>/<nome_file>.png

Scripts eseguiti per ogni dataset:
  1. plot_temp_vs_time.py      -> temperatura frame-by-frame vs tempo
  2. plot_temp_binned.py       -> temperatura mediata su bin temporali
  3. plot_temp_layer_extrap.py -> un punto per layer + fit/estrapolazione

Come funziona la sostituzione dei parametri CONFIG:
  Ogni script ha una sezione CONFIG con variabili default.
  Il runner usa regex per sostituire il valore di ogni variabile CONFIG
  direttamente nel sorgente prima di eseguirlo con exec().
  Questo garantisce che le righe DOPO la CONFIG (es. dataset_path, output_dir)
  vedano i valori aggiornati, evitando il problema della riassegnazione
  che sovrascriveva i valori iniettati nel namespace.

  NOTA TECNICA - perche' usiamo lambda nel replacement:
    re.sub(pattern, r'\1<valore>', ...) fallisce se <valore> contiene
    sequenze tipo \1, \2 ... che regex interpreta come group reference.
    Es. new_value=1.0 -> replacement=r'\11.0' -> gruppo 11 -> errore.
    La lambda evita questo: repl=lambda m: m.group(1) + str(new_value)

USO:
    python run_all_plots.py

Non serve passare argomenti: modifica solo la sezione CONFIG qui sotto.
"""

from pathlib import Path
import re
import sys
import time


# ===========================================================================
#  CONFIG  -  modifica solo questa sezione
# ===========================================================================

# ---------------------------------------------------------------------------
# Lista dei dataset da processare.
# I nomi devono corrispondere ai file .csv in creazione_dataset/datasets/
# Esempio: "ROI_wide_2_6_depth_1_2" -> usa datasets/ROI_wide_2_6_depth_1_2.csv
# ---------------------------------------------------------------------------
DATASETS = [
    "ROI_wide_2_6_depth_1_2",
    "ROI_wide_3_7_depth_1_2",
    "ROI_wide_3_10_depth_1_3",
]

# ---------------------------------------------------------------------------
# Impostazioni condivise tra tutti gli script
# ---------------------------------------------------------------------------
SHARED = {
    # Filtro validita' ROI [0.0 - 1.0]:
    # 1.0 = solo frame con la ROI completamente dentro il pezzo
    "ROI_MIN_VALID_FRAC": 1.0,

    # Includi i provini standard (senza pausa): Rec-027_std_2, Rec-G3_std_1
    "INCLUDE_STD_SPECIMENS": False,
}

# ---------------------------------------------------------------------------
# Impostazioni specifiche per plot_temp_vs_time.py
# ---------------------------------------------------------------------------
CFG_VS_TIME = {
    # Metriche da generare: "mean", "max", "min" (lista, genera un file per ognuna)
    "PLOT_METRICS": ["mean", "min", "max"],

    # Asse X: "seconds" | "frames"
    "X_AXIS": "seconds",

    # Stile linea: "linear" | "smooth"
    "LINE_STYLE": "linear",
}

# ---------------------------------------------------------------------------
# Impostazioni specifiche per plot_temp_binned.py
# ---------------------------------------------------------------------------
CFG_BINNED = {
    # Metriche da generare: "mean", "max", "min"
    "PLOT_METRICS": ["mean", "min", "max"],

    # Ampiezza bin in secondi
    "BIN_SIZE": 2,

    # Asse X: "seconds" | "frames"
    "X_AXIS": "seconds",
}

# ---------------------------------------------------------------------------
# Impostazioni specifiche per plot_temp_layer_extrap.py
# ---------------------------------------------------------------------------
CFG_EXTRAP = {
    # Metriche da generare: "mean", "max", "min"
    "PLOT_METRICS": ["mean", "min", "max"],

    # Modello di fit: "linear" | "quadratic" | "exponential"
    "FIT_MODEL": "linear",

    # Temperatura ambiente [gradi C] - solo per fit esponenziale
    "T_ENV": 25.0,

    # Mostra banda di incertezza (+-1 std per layer)
    "SHOW_SHADOW": True,
}

# ===========================================================================
#  FINE CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Path degli script di plot (stessa cartella di questo file)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent
SCRIPT_VSTIME = SCRIPT_DIR / "plot_temp_vs_time.py"
SCRIPT_BINNED = SCRIPT_DIR / "plot_temp_binned.py"
SCRIPT_EXTRAP = SCRIPT_DIR / "plot_temp_layer_extrap.py"


# ---------------------------------------------------------------------------
# Patch CONFIG via regex e poi exec()
#
# Problema risolto:
#   Iniettare variabili nel namespace prima di exec() NON funziona se
#   il sorgente riassegna quelle stesse variabili (es. DATASET_NAME = "..."
#   nella sezione CONFIG dello script figlio sovrascrive il valore iniettato).
#
# Soluzione:
#   Si legge il sorgente come testo, si sostituisce il valore stringa/numero
#   di ogni variabile CONFIG con regex, poi si esegue il sorgente patchato.
#   Cosi' le righe successive alla CONFIG (dataset_path, output_dir, ecc.)
#   vedono gia' i valori corretti.
#
# ATTENZIONE - replacement come lambda, NON come stringa r'\1<valore>':
#   Se new_value contiene cifre (es. 1.0), la stringa r'\11.0' viene
#   interpretata da re.sub come "gruppo di riferimento 11" -> errore.
#   Usando lambda m: m.group(1) + str(val) il problema non esiste.
# ---------------------------------------------------------------------------

def _patch_str_var(source: str, var_name: str, new_value: str) -> str:
    """
    Sostituisce:  VAR_NAME = "vecchio_valore"  ->  VAR_NAME = "nuovo_valore"
    Solo la prima occorrenza (la definizione in CONFIG).
    Usa lambda replacement per evitare interpretazione di backslash sequences.
    """
    pattern = rf'^({re.escape(var_name)}\s*=\s*)["\'][^"\']*["\']'
    val = new_value  # capture in closure
    return re.sub(pattern,
                  lambda m: m.group(1) + f'"{val}"',
                  source, count=1, flags=re.MULTILINE)


def _patch_bool_var(source: str, var_name: str, new_value: bool) -> str:
    """
    Sostituisce:  VAR_NAME = True/False  ->  VAR_NAME = True/False
    Usa lambda replacement per sicurezza.
    """
    val_str = "True" if new_value else "False"
    pattern = rf'^({re.escape(var_name)}\s*=\s*)(?:True|False)'
    return re.sub(pattern,
                  lambda m: m.group(1) + val_str,
                  source, count=1, flags=re.MULTILINE)


def _patch_num_var(source: str, var_name: str, new_value) -> str:
    """
    Sostituisce:  VAR_NAME = <numero>  ->  VAR_NAME = <nuovo_numero>
    Usa lambda replacement: evita che new_value (es. 1.0) venga interpretato
    come group reference (\\11 -> gruppo 11 -> re.error).
    """
    pattern = rf'^({re.escape(var_name)}\s*=\s*)[\d.]+'
    val_str = str(new_value)
    return re.sub(pattern,
                  lambda m: m.group(1) + val_str,
                  source, count=1, flags=re.MULTILINE)


def run_script_patched(script_path: Path, str_overrides: dict = None,
                       bool_overrides: dict = None,
                       num_overrides: dict = None) -> bool:
    """
    Carica il sorgente dello script, patcha le variabili CONFIG via regex,
    poi esegue il sorgente modificato con exec().

    Parametri:
      str_overrides  : {var_name: str_value}   per variabili stringa
      bool_overrides : {var_name: bool_value}  per variabili booleane
      num_overrides  : {var_name: num_value}   per variabili numeriche

    Restituisce True se OK, False se eccezione.
    """
    source = script_path.read_text(encoding="utf-8")

    for var, val in (str_overrides or {}).items():
        source = _patch_str_var(source, var, val)
    for var, val in (bool_overrides or {}).items():
        source = _patch_bool_var(source, var, val)
    for var, val in (num_overrides or {}).items():
        source = _patch_num_var(source, var, val)

    ns = {"__file__": str(script_path), "__name__": "__main__"}
    try:
        exec(compile(source, str(script_path), "exec"), ns)
        return True
    except SystemExit:
        return True
    except Exception as exc:
        print(f"    [ERRORE] {exc}")
        import traceback
        traceback.print_exc()
        return False


def close_all_figures():
    """Chiude tutte le figure matplotlib aperte (evita memory leak nei loop)."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Preview: stampa il piano di esecuzione prima di partire
# ---------------------------------------------------------------------------
def print_plan():
    n_vs  = len(CFG_VS_TIME["PLOT_METRICS"])
    n_bin = len(CFG_BINNED["PLOT_METRICS"])
    n_ext = len(CFG_EXTRAP["PLOT_METRICS"])
    n_tot = (n_vs + n_bin + n_ext) * len(DATASETS)

    print("=" * 62)
    print("  run_all_plots.py  -  piano di esecuzione")
    print("=" * 62)
    print(f"  Dataset da processare ({len(DATASETS)}):")
    for ds in DATASETS:
        print(f"    - {ds}")
    print()
    print(f"  Grafici per dataset:")
    print(f"    plot_temp_vs_time     : {n_vs} metrica/e  -> {CFG_VS_TIME['PLOT_METRICS']}")
    print(f"    plot_temp_binned      : {n_bin} metrica/e  -> {CFG_BINNED['PLOT_METRICS']}")
    print(f"    plot_temp_layer_extrap: {n_ext} metrica/e -> {CFG_EXTRAP['PLOT_METRICS']}")
    print()
    print(f"  Totale grafici attesi : {n_tot}")
    print(f"  Output dir            : Analisi_dataset/<DATASET_NAME>/")
    print()
    print(f"  Impostazioni condivise:")
    for k, v in SHARED.items():
        print(f"    {k} = {v}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    print_plan()

    confirm = input("Procedere? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes", "si", "s"):
        print("Annullato.")
        return

    print()
    total_ok  = 0
    total_err = 0
    t_start   = time.time()

    for dataset_name in DATASETS:
        print(f"{'=' * 62}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'=' * 62}")

        # Overrides condivise (bool e num)
        shared_bool = {"INCLUDE_STD_SPECIMENS": SHARED["INCLUDE_STD_SPECIMENS"]}
        shared_num  = {"ROI_MIN_VALID_FRAC":    SHARED["ROI_MIN_VALID_FRAC"]}

        # ------------------------------------------------------------------
        # 1. plot_temp_vs_time.py
        # ------------------------------------------------------------------
        for metric in CFG_VS_TIME["PLOT_METRICS"]:
            label = f"vs_time / {metric}"
            print(f"  [{label:<28}] ", end="", flush=True)
            ok = run_script_patched(
                SCRIPT_VSTIME,
                str_overrides={
                    "DATASET_NAME": dataset_name,
                    "PLOT_METRIC":  metric,
                    "X_AXIS":       CFG_VS_TIME["X_AXIS"],
                    "LINE_STYLE":   CFG_VS_TIME["LINE_STYLE"],
                },
                bool_overrides=shared_bool,
                num_overrides=shared_num,
            )
            close_all_figures()
            print("OK" if ok else "ERRORE")
            if ok:
                total_ok += 1
            else:
                total_err += 1

        # ------------------------------------------------------------------
        # 2. plot_temp_binned.py
        # ------------------------------------------------------------------
        for metric in CFG_BINNED["PLOT_METRICS"]:
            label = f"binned / {metric}"
            print(f"  [{label:<28}] ", end="", flush=True)
            ok = run_script_patched(
                SCRIPT_BINNED,
                str_overrides={
                    "DATASET_NAME": dataset_name,
                    "PLOT_METRIC":  metric,
                    "X_AXIS":       CFG_BINNED["X_AXIS"],
                },
                bool_overrides=shared_bool,
                num_overrides={**shared_num, "BIN_SIZE": CFG_BINNED["BIN_SIZE"]},
            )
            close_all_figures()
            print("OK" if ok else "ERRORE")
            if ok:
                total_ok += 1
            else:
                total_err += 1

        # ------------------------------------------------------------------
        # 3. plot_temp_layer_extrap.py
        # ------------------------------------------------------------------
        for metric in CFG_EXTRAP["PLOT_METRICS"]:
            label = f"extrap / {metric}"
            print(f"  [{label:<28}] ", end="", flush=True)
            ok = run_script_patched(
                SCRIPT_EXTRAP,
                str_overrides={
                    "DATASET_NAME": dataset_name,
                    "PLOT_METRIC":  metric,
                    "FIT_MODEL":    CFG_EXTRAP["FIT_MODEL"],
                },
                bool_overrides={
                    **shared_bool,
                    "SHOW_SHADOW": CFG_EXTRAP["SHOW_SHADOW"],
                },
                num_overrides={**shared_num, "T_ENV": CFG_EXTRAP["T_ENV"]},
            )
            close_all_figures()
            print("OK" if ok else "ERRORE")
            if ok:
                total_ok += 1
            else:
                total_err += 1

        print()

    # ------------------------------------------------------------------
    # Riepilogo finale
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("=" * 62)
    print(f"  COMPLETATO in {elapsed:.1f}s")
    print(f"  Grafici generati: {total_ok}  |  Errori: {total_err}")
    print("=" * 62)


if __name__ == "__main__":
    main()
