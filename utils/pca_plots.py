import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scree(
    pca_model, feature_names=None, figsize=(13, 5), threshold_cum=80
):
    """
    Genera un Scree Plot (varianza individual y acumulada) a partir de una
    instancia de PCA de sklearn.

    Parámetros:
    -----------
    pca_model : sklearn.decomposition.PCA
        Modelo PCA previamente ajustado (fit / fit_transform).
    feature_names : list or None, opcional
        Nombres para las etiquetas de los componentes. Si es None, usa 'PC1',
        'PC2', etc.
    figsize : tuple, opcional (default: (13, 5))
        Dimensiones de la figura.
    threshold_cum : float, opcional (default: 80)
        Línea de umbral en la varianza acumulada.
    """
    # 1. Extracción de métricas del objeto PCA de sklearn
    varianza_exp = pca_model.explained_variance_ratio_ * 100
    varianza_acum = np.cumsum(varianza_exp)
    eigenvalues = pca_model.explained_variance_
    n_components = pca_model.n_components_

    # Configuración de etiquetas para los componentes
    if feature_names is None:
        pcs = [f"PC{i + 1}" for i in range(n_components)]
    else:
        pcs = feature_names

    # 2. Construcción de la figura (tema y paleta del proyecto)
    sns.set_theme(style="white", font="sans-serif")
    color_ind, color_acum = sns.color_palette("mako", 2)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120)
    fig.suptitle(
        "Scree Plot — Varianza Explicada por Componente",
        fontsize=14,
        fontweight="bold",
        color="#1e293b",
        x=0.02,
        ha="left",
    )

    panels = [
        (
            axes[0],
            varianza_exp,
            color_ind,
            "Varianza explicada (%)",
            "Varianza por Componente",
        ),
        (
            axes[1],
            varianza_acum,
            color_acum,
            "Varianza acumulada (%)",
            "Varianza Acumulada",
        ),
    ]
    for ax, valores, color, ylabel, titulo in panels:
        ax.bar(pcs, valores, color=color, alpha=0.75, linewidth=0)
        ax.plot(
            pcs,
            valores,
            "o-",
            color="#1e293b",
            linewidth=2,
            markersize=7,
            markeredgecolor="white",
        )
        for i, v in enumerate(valores):
            ax.annotate(
                f"{v:.1f}%",
                (i, v),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#1e293b",
            )
        ax.set_ylabel(
            ylabel,
            fontsize=11,
            fontweight="bold",
            color="#475569",
            labelpad=10,
        )
        ax.set_title(
            titulo,
            fontsize=12,
            fontweight="bold",
            loc="left",
            color="#1e293b",
            pad=12,
        )
        ax.tick_params(axis="x", labelsize=9)
        # Headroom so the value labels do not touch the frame
        ax.margins(y=0.08)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
        ax.set_axisbelow(True)

    # Umbrales sobre la varianza acumulada
    axes[1].axhline(
        y=threshold_cum,
        color="#94a3b8",
        linestyle="--",
        lw=1.5,
        label=f"Umbral {threshold_cum}%",
    )
    axes[1].axhline(
        y=95, color="#ec7063", linestyle="--", lw=1.5, label="Umbral 95%"
    )
    axes[1].legend(fontsize=9, frameon=False, loc="lower right")

    sns.despine(fig=fig, top=True, right=True)
    plt.tight_layout()
    plt.show()

    # 3. Impresión de criterios de selección
    kaiser_count = (eigenvalues > 1).sum()
    print("Criterios comunes para elegir el número de componentes:")
    print(
        "  1. Umbral de varianza acumulada (80%, 90% o 95% según el contexto)"
    )
    print("  2. Regla del codo: donde la curva del scree plot 'se aplana'")
    print(
        "  3. Regla de Kaiser: conservar componentes con λ > 1 "
        "(solo aplica con datos estandarizados)"
    )
    print(
        f"\n  Regla de Kaiser en este dataset: λ > 1 → {kaiser_count} "
        "componente(s)\n"
    )

    return fig, axes


def plot_loadings(
    pca_model, feature_names, figsize=(14, 5), pc_x="PC1", pc_y="PC2"
):
    """
    Genera una figura con el heatmap de loadings de todos los componentes
    y un gráfico de barras comparativo entre dos componentes específicos.

    Parámetros:
    -----------
    pca_model : sklearn.decomposition.PCA
        Modelo PCA previamente ajustado (fit / fit_transform).
    feature_names : list or Index
        Lista con los nombres originales de las variables/características.
    figsize : tuple, opcional (default: (14, 5))
        Dimensiones de la figura.
    pc_x : str, opcional (default: 'PC1')
        Primer componente a comparar en el gráfico de barras.
    pc_y : str, opcional (default: 'PC2')
        Segundo componente a comparar en el gráfico de barras.
    """
    # 1. Construcción del DataFrame de Loadings desde la matriz components_
    # Nota: PCA de sklearn almacena los vectores propios en
    # (n_components, n_features)
    n_components = pca_model.n_components_
    pc_labels = [f"PC{i + 1}" for i in range(n_components)]

    loadings = pd.DataFrame(
        pca_model.components_.T, index=feature_names, columns=pc_labels
    )

    # 2. Construcción de la figura (tema y paletas del proyecto)
    sns.set_theme(style="white", font="sans-serif")
    color_x, color_y = sns.color_palette("mako", 2)
    cmap = sns.diverging_palette(15, 210, s=75, l=60, as_cmap=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120)
    fig.suptitle(
        "Loadings — Contribución de Variables a Componentes Principales",
        fontsize=14,
        fontweight="bold",
        color="#1e293b",
        x=0.02,
        ha="left",
    )

    # Heatmap completo
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        # Los loadings de un PCA están en [-1, 1]: escala fija entre gráficos
        vmin=-1,
        vmax=1,
        linewidths=1.5,
        linecolor="white",
        cbar=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7, "weight": "bold"},
        ax=axes[0],
    )
    axes[0].set_title(
        "Todos los componentes",
        fontsize=12,
        fontweight="bold",
        loc="left",
        color="#1e293b",
        pad=12,
    )
    axes[0].tick_params(axis="x", rotation=0, labelsize=9)

    # Gráfico de barras comparativo (PC seleccionado vs PC seleccionado)
    x = np.arange(len(feature_names))
    ancho = 0.35

    axes[1].bar(
        x - ancho / 2,
        loadings[pc_x],
        ancho,
        label=pc_x,
        color=color_x,
        alpha=0.85,
        linewidth=0,
    )
    axes[1].bar(
        x + ancho / 2,
        loadings[pc_y],
        ancho,
        label=pc_y,
        color=color_y,
        alpha=0.85,
        linewidth=0,
    )

    # Formateo de etiquetas en el eje X
    labels_formatted = [
        str(v).replace("_mm", "\n(mm)").replace("_g", "\n(g)")
        for v in feature_names
    ]

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        labels_formatted,
        fontsize=9,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes[1].axhline(0, color="#475569", lw=0.8)
    axes[1].set_ylabel(
        "Loading",
        fontsize=11,
        fontweight="bold",
        color="#475569",
        labelpad=10,
    )
    axes[1].set_title(
        f"Loadings {pc_x} vs {pc_y}",
        fontsize=12,
        fontweight="bold",
        loc="left",
        color="#1e293b",
        pad=12,
    )
    axes[1].legend(frameon=False)
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
    axes[1].set_axisbelow(True)
    sns.despine(ax=axes[1], top=True, right=True)

    plt.tight_layout()
    plt.show()

    return loadings, (fig, axes)


def plot_ppca_matrices(W, C_ppca, feature_names, figsize=(16, 6.5)):
    """
    Genera una figura con dos heatmaps del modelo PPCA: la matriz de cargas
    probabilísticas W y la covarianza implícita del modelo
    C = W Wᵀ + σ² I.

    Parámetros:
    -----------
    W : array-like, forma (p, k)
        Matriz de cargas probabilísticas (variables × componentes).
    C_ppca : array-like, forma (p, p)
        Covarianza del modelo PPCA.
    feature_names : list or Index
        Nombres de las p variables originales.
    figsize : tuple, opcional (default: (16, 6.5))
        Dimensiones de la figura.
    """
    # 1. Matrices etiquetadas (no se recalcula nada, solo se grafica)
    W = np.asarray(W)
    C_ppca = np.asarray(C_ppca)
    names = [str(v) for v in feature_names]
    pc_labels = [f"Fact_Lat{i + 1}" for i in range(W.shape[1])]
    df_w = pd.DataFrame(W, index=names, columns=pc_labels)
    df_c = pd.DataFrame(C_ppca, index=names, columns=names)

    # 2. Construcción de la figura (tema y paletas del proyecto)
    sns.set_theme(style="white", font="sans-serif")
    cmap = sns.diverging_palette(15, 210, s=75, l=60, as_cmap=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=120,
        gridspec_kw={"width_ratios": [W.shape[1] + 2, len(names) + 2]},
    )
    fig.suptitle(
        "PPCA — Cargas Probabilísticas y Covarianza del Modelo",
        fontsize=14,
        fontweight="bold",
        color="#1e293b",
        x=0.02,
        ha="left",
    )

    panels = [
        (axes[0], df_w, "Cargas probabilísticas W"),
        (axes[1], df_c, "Covarianza del modelo C = WWᵀ + σ²I"),
    ]
    for ax, matriz, titulo in panels:
        # W no está acotada en [-1, 1]: escala simétrica según los datos
        limite = float(np.abs(matriz.to_numpy()).max())
        sns.heatmap(
            matriz,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-limite,
            vmax=limite,
            linewidths=1.5,
            linecolor="white",
            cbar=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(
            titulo,
            fontsize=12,
            fontweight="bold",
            loc="left",
            color="#1e293b",
            pad=12,
        )
        ax.tick_params(axis="both", labelsize=9)

    axes[0].tick_params(axis="x", rotation=0)
    plt.setp(
        axes[1].get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    plt.tight_layout()
    plt.show()

    return fig, axes
