"""
Panel de diagnóstico de normalidad (histograma + boxplot + QQ + test).
Estilo editorial flat, optimizado para datasets grandes.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

# --- PALETA / CONSTANTES DE ESTILO ---
C_DARK = "#2c3e50"
C_MID = "#34495e"
C_SOFT = "#7f8c8d"
C_BLUE = "#5dade2"
C_GREY = "#95a5a6"
C_ORANGE = "#e67e22"
C_GRID = "#eeeeee"
C_BOX = "#f8f9f9"

# --- CONSTANTES ESTADÍSTICAS ---
#: Tests admitidos por :func:`normality_report`.
TESTS = ("auto", "shapiro", "dagostino", "ks")
#: Estrategias admitidas para x <= 0 bajo log.
NEG_STRATEGIES = ("shift", "signed", "drop")
#: Por encima de este n la aproximación del p-valor de Shapiro-Wilk
#: deja de ser fiable (scipy emite su propio aviso).
SHAPIRO_MAX_N = 5_000
#: statsmodels trunca el p-valor de Lilliefors por debajo de este
#: valor: la tabla no llega más abajo, así que no es una medición.
LILLIEFORS_P_FLOOR = 1e-3
#: Proporción de valores únicos por debajo de la cual la muestra se
#: considera un retículo (empates masivos) para los tests continuos.
TIES_UNIQUE_RATIO = 0.8
#: Mínimo de observaciones para que el panel tenga sentido.
MIN_N = 8


# ----------------------------------------------------------------------
# Helpers internos (numpy puro = rápido)
# ----------------------------------------------------------------------
def _log_transform(
    x: np.ndarray, neg_strategy: str
) -> tuple[np.ndarray, str, int]:
    """Aplica log10 gestionando ceros y negativos.

    La estrategia se elige por su nombre, no por el signo del mínimo:
    ``'drop'`` descarta todo ``x <= 0``, haya negativos o solo ceros.
    Tratar los ceros aparte (mandándolos a ``log10(0+1) = 0``) los
    convierte en outliers artificiales que invierten la asimetría.

    Returns
    -------
    tuple
        ``(x_transformado, etiqueta, n_descartados)``.
    """
    if neg_strategy not in NEG_STRATEGIES:
        raise ValueError(
            f"neg_strategy debe ser uno de {NEG_STRATEGIES}, "
            f"no {neg_strategy!r}."
        )

    mn = x.min()
    if mn > 0:
        return np.log10(x), "log(x)", 0

    if neg_strategy == "drop":
        pos = x[x > 0]
        n_out = int(x.size - pos.size)
        if pos.size < MIN_N:
            raise ValueError(
                f"Quedan {pos.size} valores positivos tras 'drop' "
                f"(mínimo {MIN_N})."
            )
        return np.log10(pos), "log(x), x>0", n_out

    if neg_strategy == "signed":
        # Log simétrico: conserva el signo y el orden.
        return (
            np.sign(x) * np.log10(1.0 + np.abs(x)),
            "sign(x)·log(1+|x|)",
            0,
        )

    # 'shift': traslada la distribución al dominio positivo.
    label = "log(x + 1)" if mn == 0 else f"log(x - min + 1)  [min={mn:,.4g}]"
    return np.log10(x - mn + 1.0), label, 0


def _fd_bins(x: np.ndarray, q1: float, q3: float, max_bins: int = 120) -> int:
    """Regla Freedman-Diaconis con tope (evita miles de bins inútiles)."""
    iqr = q3 - q1
    rng = x[-1] - x[0]  # x viene ordenado
    if iqr <= 0 or rng <= 0:
        return int(np.clip(np.sqrt(x.size), 10, max_bins))
    h = 2.0 * iqr * x.size ** (-1 / 3)
    return int(np.clip(np.ceil(rng / h), 10, max_bins))


def _warn_ties(x: np.ndarray, nombre: str) -> None:
    """Avisa si hay tantos empates que el test continuo pierde sentido."""
    n_unique = np.unique(x).size
    if n_unique < TIES_UNIQUE_RATIO * x.size:
        warnings.warn(
            f"{nombre} supone una distribución continua, pero la "
            f"muestra tiene {n_unique:,} valores distintos en "
            f"{x.size:,} observaciones. Con tantos empates el "
            f"estadístico se sesga y el p-valor no es interpretable.",
            stacklevel=3,
        )


def _normality_test(x: np.ndarray, test: str) -> tuple[str, float, float, int]:
    """Elige el test según el tamaño muestral de ESTA muestra.

    Cada muestra resuelve ``'auto'`` con su propio n, de modo que el
    contraste sobre ``x`` y el contraste sobre ``log(x)`` pueden acabar
    usando tests distintos sobre tamaños distintos. Es intencionado: por
    eso se devuelve también el ``n`` y el panel lo imprime, para que la
    diferencia sea visible y no se comparen p-valores incomparables.

    Returns
    -------
    tuple
        ``(nombre, estadístico, p_valor, n)``.
    """
    n = int(x.size)
    if test == "auto":
        test = "shapiro" if n <= SHAPIRO_MAX_N else "dagostino"

    if test == "shapiro":
        # El tope de 5000 es de la aproximación del p-valor, no del
        # coste: el algoritmo es O(n log n). scipy avisa por su cuenta.
        _warn_ties(x, "Shapiro-Wilk")
        s, p = stats.shapiro(x)
        return "Shapiro-Wilk", float(s), float(p), n

    if test == "dagostino":  # asimetría + curtosis, barato en n grande
        s, p = stats.normaltest(x)
        return "D'Agostino-Pearson K²", float(s), float(p), n

    if test == "ks":
        sd = x.std(ddof=1)
        if sd <= 0 or not np.isfinite(sd):
            return "Lilliefors (KS)", np.nan, np.nan, n
        _warn_ties(x, "Lilliefors (KS)")
        # lilliefors estima media y sd internamente y aplica la
        # corrección por parámetros estimados.
        s, p = lilliefors(x, dist="norm")
        return "Lilliefors (KS)", float(s), float(p), n

    raise ValueError(f"test debe ser uno de {TESTS}, no {test!r}.")


def _fmt_block(
    nombre: str,
    stat: float,
    p: float,
    n: int,
    etiqueta: str,
    alpha: float,
) -> tuple[bool | None, str]:
    """Bloque de texto de un contraste. Devuelve ``(veredicto, texto)``.

    ``veredicto`` es ``None`` cuando no hay p-valor: sin él no se puede
    ni rechazar ni dejar de rechazar H0, y darlo por rechazado (que es
    lo que hace ``nan > alpha``) sería inventar una conclusión.
    """
    cab = f"{etiqueta}\n{nombre} · n = {n:,}"
    stat_s = "n/d" if not np.isfinite(stat) else f"{stat:,.5g}"

    if p is None or not np.isfinite(p):
        return None, (
            f"{cab}\nestadístico = {stat_s} · p = n/d\n"
            f"Resultado no concluyente"
        )

    if nombre.startswith("Lilliefors") and p <= LILLIEFORS_P_FLOOR:
        # Suelo de la tabla de statsmodels, no un valor medido.
        ps = f"< {LILLIEFORS_P_FLOOR:g}"
    elif p <= 0.0:
        # p = 0 exacto es desbordamiento a cero, no una medición.
        ps = "< 1e-300"
    elif p >= 1e-4:
        ps = f"= {p:.4f}"
    else:
        ps = f"= {p:.2e}"

    ok = bool(p > alpha)
    ver = "NO se rechaza H0" if ok else "Se rechaza H0"
    return ok, (f"{cab}\nestadístico = {stat_s} · p {ps}\nα = {alpha} → {ver}")


# ----------------------------------------------------------------------
# Función principal
# ----------------------------------------------------------------------
def normality_report(
    df: pd.DataFrame,
    col: str,
    log_scale: bool = False,
    *,
    neg_strategy: str = "shift",  # 'shift' | 'signed' | 'drop'
    test: str = "auto",  # 'auto' | 'shapiro' | 'dagostino' | 'ks'
    alpha: float = 0.05,
    color: str = C_BLUE,
    kde: bool = True,
    max_qq_points: int = 5_000,  # puntos dibujados en el QQ
    max_fliers: int = 2_000,  # outliers dibujados en el boxplot
    kde_max_n: int = 20_000,  # a partir de aquí la KDE se muestrea
    figsize: tuple[float, float] = (17, 5.6),
    random_state: int = 0,
    show: bool = True,
) -> dict:
    """
    Panel 1x3 (histograma | boxplot | QQ-plot) + test de normalidad.

    Parameters
    ----------
    df : DataFrame de origen.
    col : nombre de la columna numérica.
    log_scale : aplica log₁₀ a los datos antes de analizarlos.
    neg_strategy : cómo tratar valores <= 0 cuando log_scale=True.
        'shift'  -> log₁₀(x - min + 1)  (conserva todas las filas)
        'signed' -> sign(x)·log₁₀(1+|x|) (log simétrico en torno a 0)
        'drop'   -> descarta x <= 0 (ceros incluidos)
    test : test de normalidad. 'auto' usa Shapiro-Wilk si n<=5000, si no
        K². Con log_scale se ejecutan DOS contrastes independientes —uno
        sobre x y otro sobre la transformada— y cada uno resuelve 'auto'
        con su propio n, así que pueden ser tests distintos. El panel
        imprime nombre, n y estadístico de cada uno.

    Returns
    -------
    dict
        ``n``, ``n_dropped``, ``n_dropped_transform``, ``media``,
        ``mediana``, ``std``, ``skew``, ``kurtosis``, ``transform``,
        ``test``, ``statistic``, ``p_value``, ``is_normal``, y los
        homólogos ``*_raw`` del contraste sobre x sin transformar
        (``None``/``nan`` si ``log_scale=False``), más ``fig``.
    """
    if col not in df.columns:
        raise KeyError(f"'{col}' no está en el DataFrame.")
    if test not in TESTS:
        raise ValueError(f"test debe ser uno de {TESTS}, no {test!r}.")
    if log_scale and neg_strategy not in NEG_STRATEGIES:
        raise ValueError(
            f"neg_strategy debe ser uno de {NEG_STRATEGIES}, "
            f"no {neg_strategy!r}."
        )

    # --- 0. DATOS: una sola pasada, numpy contiguo float64 ---
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    n_raw_total = x.size
    x = x[np.isfinite(x)]
    n_dropped = n_raw_total - x.size
    if x.size < MIN_N:
        raise ValueError(
            f"'{col}' tiene solo {x.size} valores válidos (mínimo {MIN_N})."
        )

    transform = "identidad"
    x_raw = None
    n_dropped_t = 0
    if log_scale:
        x_raw = x
        x, transform, n_dropped_t = _log_transform(x, neg_strategy)
        x = x[np.isfinite(x)]
        if x.size < MIN_N:
            raise ValueError(
                f"'{col}' queda con {x.size} valores finitos tras "
                f"'{transform}' (mínimo {MIN_N})."
            )

    # Ordenar UNA vez: sirve para cuantiles, boxplot y QQ (O(n log n))
    xs = np.sort(x)
    n = xs.size
    q1, med, q3 = np.percentile(xs, [25, 50, 75], method="linear")
    iqr = q3 - q1
    mean, std = xs.mean(), xs.std(ddof=1)
    degenerate = not np.isfinite(std) or std <= 0  # columna constante
    # Estimadores insesgados; sin varianza no tienen sentido.
    skew, kurt = (
        (np.nan, np.nan)
        if degenerate
        else (
            float(stats.skew(xs, bias=False)),
            float(stats.kurtosis(xs, bias=False)),
        )
    )

    # --- 1. CONFIGURACIÓN (aislada: no muta el estado global) ---
    with plt.rc_context():
        sns.set_theme(style="white", context="notebook")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["text.color"] = "#333333"

        # --- 2. CREAR FIGURA ---
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax_hist, ax_box, ax_qq = axes

        unit = f"{col} [{transform}]" if log_scale else col
        fig.text(
            0.06,
            0.955,
            f"Diagnóstico de Distribución: {col}",
            fontsize=20,
            fontweight="bold",
            color=C_DARK,
        )
        escala = "logarítmica" if log_scale else "original"
        sub = f"Escala {escala} · n = {n:,}"
        if n_dropped:
            sub += f" · {n_dropped:,} nulos descartados"
        if n_dropped_t:
            sub += f" · {n_dropped_t:,} no positivos descartados"
        if log_scale:
            sub += f" · transformación: {transform}"
        fig.text(0.06, 0.905, sub, fontsize=12.5, color=C_SOFT)

        # ---------- PANEL 1: HISTOGRAMA ----------
        ax_hist.grid(
            axis="y", color=C_GRID, linestyle="-", linewidth=1, zorder=0
        )
        bins = _fd_bins(xs, q1, q3)

        # KDE: se estima sobre una submuestra si n es grande. Ojo: eso
        # ensancha el ancho de banda (Scott ∝ n^-1/5) y sobre-suaviza
        # ligeramente respecto al histograma, que sí usa todo el dato.
        kde_data, use_kde = xs, kde and not degenerate
        if use_kde and n > kde_max_n:
            rng = np.random.default_rng(random_state)
            kde_data = rng.choice(xs, kde_max_n, replace=False)

        # np.histogram cuenta en C; seaborn solo recibe `bins` puntos
        counts, edges = np.histogram(xs, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # `bins` como lista: seaborn compara bins == "auto" y falla
        # con un ndarray.
        sns.histplot(
            x=centers,
            weights=counts,
            bins=edges.tolist(),
            ax=ax_hist,
            color=color,
            edgecolor="none",
            alpha=0.85,
            stat="density",
            zorder=2,
        )
        if use_kde:
            sns.kdeplot(
                x=kde_data,
                ax=ax_hist,
                color=C_DARK,
                linewidth=1.6,
                zorder=3,
            )

        ax_hist.axvline(
            med,
            color=C_DARK,
            ls="--",
            lw=1.4,
            zorder=4,
            label=f"Mediana: {med:,.3g}",
        )
        ax_hist.axvline(
            mean,
            color=C_ORANGE,
            ls=":",
            lw=1.8,
            zorder=4,
            label=f"Media: {mean:,.3g}",
        )
        ax_hist.legend(frameon=False, fontsize=9.5, loc="upper right")

        ax_hist.set_title(
            "Histograma y Densidad",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_hist.set_xlabel(unit, fontsize=11, fontweight="medium")
        ax_hist.set_ylabel("Densidad", fontsize=11, fontweight="medium")

        # ---------- PANEL 2: BOXPLOT ----------
        ax_box.grid(
            axis="x", color=C_GRID, linestyle="-", linewidth=1, zorder=0
        )

        # xs ordenado -> bigotes y outliers por búsqueda binaria
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        i_lo = int(np.searchsorted(xs, lo, side="left"))
        i_hi = int(np.searchsorted(xs, hi, side="right"))
        whislo = xs[i_lo] if i_lo < n else xs[0]
        whishi = xs[i_hi - 1] if i_hi > 0 else xs[-1]
        out = np.concatenate([xs[:i_lo], xs[i_hi:]])
        n_out = out.size

        # bxp dibuja desde el resumen ya calculado => coste O(1) en n
        ax_box.bxp(
            [
                dict(
                    label="",
                    med=med,
                    q1=q1,
                    q3=q3,
                    whislo=whislo,
                    whishi=whishi,
                    fliers=[],
                )
            ],
            positions=[0],
            orientation="horizontal",
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, edgecolor="none", alpha=0.85),
            medianprops=dict(color=C_DARK, linewidth=2),
            whiskerprops=dict(color=C_GREY, linewidth=1.3),
            capprops=dict(color=C_GREY, linewidth=1.3),
            zorder=2,
        )
        ax_box.set_ylim(-0.6, 0.6)
        if n_out:
            draw = out
            if n_out > max_fliers:
                rng = np.random.default_rng(random_state)
                draw = rng.choice(out, max_fliers, replace=False)
            jitter = np.random.default_rng(random_state).uniform(
                -0.06, 0.06, draw.size
            )
            ax_box.scatter(
                draw,
                jitter,
                s=12,
                color=C_GREY,
                alpha=0.45,
                edgecolors="none",
                zorder=3,
            )

        ax_box.scatter([mean], [0], marker="D", s=45, color=C_ORANGE, zorder=5)
        ax_box.set_title(
            "Dispersión y Outliers",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_box.set_xlabel(unit, fontsize=11, fontweight="medium")
        ax_box.set_yticks([])

        box_txt = f"IQR {iqr:,.4g}\nOutliers  {n_out:,} ({n_out / n:.1%})"
        ax_box.text(
            0.02,
            0.04,
            box_txt,
            transform=ax_box.transAxes,
            fontsize=9.5,
            va="bottom",
            ha="left",
            color=C_MID,
            linespacing=1.5,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=C_BOX,
                edgecolor="#dddddd",
            ),
        )

        # ---------- PANEL 3: QQ-PLOT ----------
        ax_qq.grid(color=C_GRID, linestyle="-", linewidth=1, zorder=0)

        # Submuestreo por cuantiles: mantiene la forma exacta de las
        # colas y respeta max_qq_points (dibujar n puntos en un dataset
        # de 10^5 no aporta nada y cuesta segundos).
        if n > max_qq_points:
            idx = np.unique(
                np.linspace(0, n - 1, max_qq_points).astype(np.int64)
            )
        else:
            idx = np.arange(n)
        sample_q = xs[idx]
        probs = (idx + 0.5) / n  # posiciones de Hazen
        theor_q = stats.norm.ppf(probs)

        ax_qq.scatter(
            theor_q,
            sample_q,
            s=11,
            color=color,
            alpha=0.6,
            edgecolors="none",
            zorder=2,
        )

        # Recta de referencia por cuartiles (equivale a line="q"):
        # robusta a las colas, a diferencia de la recta media/sd.
        tq1, tq3 = stats.norm.ppf([0.25, 0.75])
        if iqr > 0:
            slope = iqr / (tq3 - tq1)
            intercept = q1 - slope * tq1
            ref_x = np.array([theor_q[0], theor_q[-1]])
            ax_qq.plot(
                ref_x,
                intercept + slope * ref_x,
                color=C_ORANGE,
                linewidth=1.8,
                zorder=3,
                label="Referencia normal",
            )
            ax_qq.legend(frameon=False, fontsize=9.5, loc="upper left")

        ax_qq.set_title(
            "Q-Q Plot Normal",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_qq.set_xlabel(
            "Cuantiles teóricos", fontsize=11, fontweight="medium"
        )
        ax_qq.set_ylabel(
            f"Cuantiles observados · {unit}",
            fontsize=11,
            fontweight="medium",
        )

        # --- TESTS DE HIPÓTESIS ---
        # H0: la muestra procede de una distribución normal.
        name_raw: str | None = None
        statistic_raw = p_value_raw = np.nan
        n_test_raw = 0
        is_normal_raw: bool | None = None
        info_raw = ""

        if degenerate:
            name, statistic, p_value = "sin varianza", np.nan, np.nan
            n_test, is_normal = n, None
            info = "Columna constante\nEl test de normalidad no es aplicable"
        else:
            name, statistic, p_value, n_test = _normality_test(xs, test)
            etiqueta = (
                f"H0: {transform} es normal"
                if log_scale
                else "H0: los datos son normales"
            )
            is_normal, info = _fmt_block(
                name, statistic, p_value, n_test, etiqueta, alpha
            )

            if log_scale:
                # Contraste independiente sobre x sin transformar: elige
                # su propio test según SU n, que con 'drop' ni siquiera
                # es el mismo. Por eso cada bloque lleva su n.
                (
                    name_raw,
                    statistic_raw,
                    p_value_raw,
                    n_test_raw,
                ) = _normality_test(x_raw, test)
                is_normal_raw, info_raw = _fmt_block(
                    name_raw,
                    statistic_raw,
                    p_value_raw,
                    n_test_raw,
                    "H0: x (sin transformar) es normal",
                    alpha,
                )

        edge = (
            C_GREY
            if is_normal is None
            else (C_BLUE if is_normal else C_ORANGE)
        )
        ax_qq.text(
            0.97,
            0.05,
            info + (f"\n\n{info_raw}" if info_raw else ""),
            transform=ax_qq.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="right",
            color=C_MID,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=C_BOX,
                edgecolor=edge,
                linewidth=1.4,
            ),
        )

        # --- ESTÉTICA FINAL ---
        for a in axes:
            sns.despine(ax=a, left=True, bottom=True)
            a.tick_params(colors=C_SOFT, labelsize=9.5)

        plt.subplots_adjust(
            top=0.78,
            left=0.06,
            right=0.97,
            bottom=0.13,
            wspace=0.25,
        )
        if show:
            plt.show()

    # Se cierra siempre: el notebook itera sobre columnas y matplotlib
    # no libera las figuras solo. `fig` sigue admitiendo savefig().
    plt.close(fig)

    return {
        "n": int(n),
        "n_dropped": int(n_dropped),
        "n_dropped_transform": int(n_dropped_t),
        "media": float(mean),
        "mediana": float(med),
        "std": float(std),
        "skew": skew,
        "kurtosis": kurt,
        "transform": transform,
        "test": name,
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": is_normal,
        "n_test": int(n_test),
        "test_raw": name_raw,
        "statistic_raw": statistic_raw,
        "p_value_raw": p_value_raw,
        "is_normal_raw": is_normal_raw,
        "n_raw": int(n_test_raw),
        "fig": fig,
    }
