from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

GRANO = {
    "Area": 30000,
    "Perimeter": 6000,
    "MajorAxisLength": 230,
    "MinorAxisLength": 140,
    "AspectRation": 2,
    "Eccentricity": 0.6,
    "ConvexArea": 31000,
    "EquivDiameter": 200,
    "Extent": 0.7,
    "Solidity": 0.9,
    "roundness": 0.8,
    "Compactness": 0.7,
    "ShapeFactor1": 0.009,
    "ShapeFactor2": 0.002,
    "ShapeFactor3": 0.8,
    "ShapeFactor4": 1,
}


def main():
    X = pd.DataFrame([GRANO])
    ovr = joblib.load(ROOT / "results" / "modelo_svm_OvR.joblib")
    ovo = joblib.load(ROOT / "results" / "modelo_svm_OvO.joblib")
    classes = ovr.classes_

    ovr_scores = ovr.decision_function(X)[0]
   
    ovo_scores = ovo.decision_function(X)[0]
    ovo_votes = np.rint(ovo_scores).astype(int)

    tabla = pd.DataFrame(
        {"OvR puntuación": ovr_scores, "OvO votos (de 6)": ovo_votes},
        index=classes,
    ).sort_values("OvO votos (de 6)", ascending=False)
    print(tabla.to_string(float_format="%.3f"))

    pred_ovr, pred_ovo = ovr.predict(X)[0], ovo.predict(X)[0]
    print(f"\nPredicción OvR: {pred_ovr}")
    print(f"Predicción OvO: {pred_ovo}")
    print("→ Coinciden" if pred_ovr == pred_ovo else "→ NO coinciden")


if __name__ == "__main__":
    main()
