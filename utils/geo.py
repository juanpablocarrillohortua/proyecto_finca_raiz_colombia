"""Geographic helpers for the FincaRaiz EDA."""
from __future__ import annotations
import numpy as np



from typing import Any, NamedTuple

import pandas as pd


class BBox(NamedTuple):
    """Axis-aligned lat/lon box, decimal degrees (WGS84)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


#: The whole Cundinamarca department, rounded outward. Bogota D.C. sits
#: inside it, so Soacha, Chia and the rest of the savanna municipalities
#: are all admitted. A rectangle, not the real departmental boundary: it
#: also covers slices of Boyaca, Tolima and Meta. Deliberately generous
#: -- it is here to catch coordinates that are plainly broken, not to
#: decide which municipality a listing belongs to.
CUNDINAMARCA_BBOX = BBox(
    lat_min=3.65,  # Cabrera, southern tip
    lat_max=5.85,  # northern border with Boyaca
    lon_min=-75.00,  # Magdalena river, western border
    lon_max=-73.00,  # Medina, eastern border with Meta/Casanare
)


def inside_bbox_mask(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> pd.Series:
    """Flag the points that fall inside ``bbox``, edges included.

    Rows with a missing latitude or longitude come back ``False``: a
    null is not evidence of being inside the box.
    """
    return lat.between(bbox.lat_min, bbox.lat_max) & lon.between(
        bbox.lon_min, bbox.lon_max
    )


def single_bbox_inside(lat: int, lon: int, bbox: BBox) -> bool:
    """Check a single one coordinate"""

    return (bbox.lat_min < lat < bbox.lat_max) and (
        bbox.lon_min < lon < bbox.lon_max
    )


def outside_bbox_mask(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> pd.Series:
    """Flag the points that fall outside ``bbox``.

    Not simply ``~inside_bbox_mask(...)``. ``Series.between`` returns
    ``False`` for ``NaN``, so negating it would report every missing
    coordinate as outside the box. Nulls are excluded here and counted
    separately by :func:`count_outside_bogota`.
    """
    known = lat.notna() & lon.notna()
    return ~inside_bbox_mask(lat, lon, bbox) & known


def count_outside_bogota(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> dict[str, Any]:
    """Summarise how many points miss the Bogota region.

    The default box spans the whole Cundinamarca department, so a point
    is only flagged when its coordinates are implausible for the region
    altogether -- not when it merely sits in a neighbouring municipality
    such as Soacha, Chia or Zipaquira, all of which count as inside.

    Returns ``total``, ``inside``, ``outside``, ``missing``,
    ``pct_outside`` (share of ``total``, 2 dp) and the ``bbox`` used.
    ``inside + outside + missing`` always equals ``total``.
    """
    total = len(lat)
    missing = int((lat.isna() | lon.isna()).sum())
    outside = int(outside_bbox_mask(lat, lon, bbox).sum())
    return {
        "total": total,
        "inside": total - outside - missing,
        "outside": outside,
        "missing": missing,
        "pct_outside": round(100 * outside / total, 2) if total else 0.0,
        "bbox": bbox,
    }

def add_landmark_distances(
    df_houses: pd.DataFrame,
    df_landmarks: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Calcula y agrega a `df_houses` la distancia (en km) al landmark positivo

    y negativo más cercano.
    """
    # Separar landmarks por tipo
    pos_landmarks = df_landmarks[df_landmarks["label"] == "positive"][
        [lat_col, lon_col]
    ].to_numpy()
    neg_landmarks = df_landmarks[df_landmarks["label"] == "negative"][
        [lat_col, lon_col]
    ].to_numpy()

    # Coordenadas de las viviendas
    houses_coords = df_houses[[lat_col, lon_col]].to_numpy()

    def min_haversine_distances(coords_a, coords_b):
        """Distancia mínima en km de cada punto de A al conjunto B.

        usando broadcasting de NumPy.
        """
        earth_radius_km = 6371.0

        # Convertir a radianes
        lat1, lon1 = np.radians(coords_a[:, 0]), np.radians(coords_a[:, 1])
        lat2, lon2 = np.radians(coords_b[:, 0]), np.radians(coords_b[:, 1])

        # Matriz 2D: Filas = Viviendas, Columnas = Landmarks
        dlat = lat1[:, np.newaxis] - lat2
        dlon = lon1[:, np.newaxis] - lon2

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1)[:, np.newaxis]
            * np.cos(lat2)
            * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Retorna la distancia mínima (al landmark más cercano) por cada
        # vivienda
        return np.min(earth_radius_km * c, axis=1)

    # Crear una copia del DataFrame para no modificar el original directamente
    df_result = df_houses.copy()

    # Asignar las 2 nuevas columnas
    df_result["dist_nearest_positive_km"] = min_haversine_distances(
        houses_coords, pos_landmarks
    )
    df_result["dist_nearest_negative_km"] = min_haversine_distances(
        houses_coords, neg_landmarks
    )

    return df_result