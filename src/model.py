# src/model.py
"""
Módulo de modelo para CancelGuard.

Entrenamos un árbol de decisión a partir de los datos cada vez que se
arranca la aplicación (cuando se llama a load_model en app.py) y
definimos la función de inferencia predict_cancellation.

El modelo solo utiliza 4 variables numéricas:
- lead_time
- total_nights
- adr
- total_of_special_requests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import etl


@dataclass
class CancelGuardModel:
    """
    Contenedor del árbol de decisión entrenado.
    """
    tree: DecisionTreeClassifier
    feature_names: list[str]


def load_model() -> CancelGuardModel:
    """
    Entrena (o reentrena) un árbol de decisión a partir de los datos.

    Se llama una sola vez al arrancar la app en app.py, y luego el
    modelo se reutiliza en los callbacks.
    """
    df = etl.load_data()

    # Aseguramos que existe total_nights
    if "total_nights" not in df.columns and {
        "stays_in_weekend_nights",
        "stays_in_week_nights",
    }.issubset(df.columns):
        df["total_nights"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )

    required_cols = [
        "lead_time",
        "total_nights",
        "adr",
        "total_of_special_requests",
        "is_canceled",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas necesarias para entrenar el modelo: {missing}"
        )

    # Dataset para el modelo
    df_model = df[required_cols].dropna()

    X = df_model[["lead_time", "total_nights", "adr", "total_of_special_requests"]]
    y = df_model["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Árbol un poco más complejo y equilibrado
    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train, y_train)

    # Si quieres ver el rendimiento:
    # acc = tree.score(X_test, y_test)
    # print(f"Accuracy del árbol de decisión: {acc:.3f}")

    return CancelGuardModel(tree=tree, feature_names=list(X.columns))


def predict_cancellation(model: CancelGuardModel, data: Dict[str, Any]):
    """
    Recibe un diccionario con las características de la reserva y devuelve:
    - pred (0/1)
    - prob (float en [0,1])

    Usa el árbol de decisión entrenado en load_model().
    """
    if model is None or not isinstance(model, CancelGuardModel):
        raise ValueError(
            "El modelo no está inicializado correctamente. "
            "Asegúrate de llamar a load_model() en app.py."
        )

    # Extraemos variables numéricas con valores por defecto
    lead_time = float(data.get("lead_time") or 0)
    total_nights = float(data.get("total_nights") or 1)
    adr = float(data.get("adr") or 0.0)
    special_requests = float(data.get("total_of_special_requests") or 0)

    features = np.array(
        [[lead_time, total_nights, adr, special_requests]],
        dtype=float,
    )

    proba = model.tree.predict_proba(features)[0, 1]
    pred = int(proba > 0.5)

    return pred, float(proba)
