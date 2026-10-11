import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


def _wall(x0, y, height, color):
    """Vertical sheet at x = x0 under the curve ``height`` over ``y``."""
    n = len(y)
    low = np.arange(n - 1)
    return go.Mesh3d(
        x=np.full(2 * n, x0),
        y=np.tile(y, 2),
        z=np.concatenate([np.zeros(n), height]),
        # Two triangles per grid step, between the floor row and the top row.
        i=np.tile(low, 2),
        j=np.concatenate([low + 1, low + n + 1]),
        k=np.concatenate([low + n + 1, low + n]),
        color=color,
        opacity=0.75,
        hoverinfo="skip",
    )


def _edge(x0, y, height, color):
    """Top edge of the wall: the sliced curve itself."""
    return go.Scatter3d(
        x=np.full(len(y), x0),
        y=y,
        z=height,
        mode="lines",
        line={"color": color, "width": 7},
        hovertemplate="%{y:.4g}<br>%{z:.3g}<extra></extra>",
    )


def _number(value):
    """Slider text: plain digits for large values, never 1.2e+06."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def conditional_plot(
    df,
    given_col,
    free_col,
    title=None,
    colorscale="Tealgrn",
    grid_size=60,
    n_steps=25,
):
    """Interactive 3D view of f(y | x): the slice of the joint density at
    X = x0, and the same slice normalised by f_X(x0).

    ``title`` replaces the default figure title."""
    data = df[[given_col, free_col]].dropna()
    given_label = given_col.replace("_", " ").title()
    free_label = free_col.replace("_", " ").title()

    # The far tails of the given variable are left out: there f_X is close
    # to zero and the ratio joint / f_X is numerically meaningless.
    x = np.linspace(*data[given_col].quantile([0.02, 0.98]), n_steps)
    kde = gaussian_kde(data.to_numpy().T)
    # The free variable is padded by three kernel widths: the estimate
    # spreads that far past the data, and without it a slice would not
    # integrate to its full area.
    spill = 3 * kde.factor * data[free_col].std()
    y = np.linspace(
        data[free_col].min() - spill, data[free_col].max() + spill, grid_size
    )
    grid_x, grid_y = np.meshgrid(x, y)
    joint = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    joint = joint.reshape(grid_x.shape)
    # Same bandwidth factor as the joint estimate, so this is exactly its
    # marginal: the area of each slice of the joint surface.
    f_x = gaussian_kde(data[given_col], bw_method=kde.factor)(x)
    cond = joint / f_x

    color_cond, color_joint = sample_colorscale(colorscale, [0.95, 0.6])
    start = n_steps // 2

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
        subplot_titles=(
            "<b>Densidad condicional</b><br>"
            "<sup>la rebanada dividida por f<sub>X</sub>(x₀): área = 1</sup>",
            "<b>Rebanada de la densidad conjunta</b><br>"
            "<sup>corte en X = x₀: área = f<sub>X</sub>(x₀)</sup>",
        ),
    )
    surface = {"colorscale": colorscale, "opacity": 0.4, "showscale": False}
    hover = f"{given_label}: %{{x:.4g}}<br>{free_label}: %{{y:.4g}}<br>"
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=cond,
            hovertemplate=hover + "f(y | x): %{z:.3g}<extra></extra>",
            **surface,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=joint,
            hovertemplate=hover + "f(x, y): %{z:.3g}<extra></extra>",
            **surface,
        ),
        row=1,
        col=2,
    )
    # The four traces the slider moves: a wall and its top edge per panel.
    fig.add_trace(_wall(x[start], y, cond[:, start], color_cond), row=1, col=1)
    fig.add_trace(_edge(x[start], y, cond[:, start], color_cond), row=1, col=1)
    fig.add_trace(
        _wall(x[start], y, joint[:, start], color_joint), row=1, col=2
    )
    fig.add_trace(
        _edge(x[start], y, joint[:, start], color_joint), row=1, col=2
    )

    def status(k):
        return (
            f"{given_label} = x₀ = {_number(x[k])}   ·   "
            f"f<sub>X</sub>(x₀) = {f_x[k]:.3g}   ·   "
            "área de la rebanada normalizada = 1"
        )

    n = len(y)
    steps = [
        {
            "method": "update",
            "label": _number(x[k]),
            "args": [
                {
                    "x": [np.full(2 * n, x[k]), np.full(n, x[k])] * 2,
                    "z": [
                        np.concatenate([np.zeros(n), cond[:, k]]),
                        cond[:, k],
                        np.concatenate([np.zeros(n), joint[:, k]]),
                        joint[:, k],
                    ],
                },
                {"annotations[3].text": status(k)},
                [2, 3, 4, 5],
            ],
        }
        for k in range(n_steps)
    ]

    fig.add_annotation(
        text=(
            "<b>f<sub>Y|X</sub>(y|x) = "
            "f<sub>X,Y</sub>(x,y) / f<sub>X</sub>(x)</b>"
        ),
        x=0.5,
        y=1.13,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 17},
    )
    fig.add_annotation(
        text=status(start),
        x=0.5,
        y=-0.04,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13},
    )

    def scene(z_title, z_max):
        return {
            "xaxis": {"title": given_label},
            "yaxis": {"title": free_label},
            # Fixed range, so the scene does not rescale while sliding.
            "zaxis": {
                "title": z_title,
                "range": [0, 1.05 * z_max],
                "exponentformat": "e",
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.8},
            "camera": {
                "eye": {"x": 1.3, "y": 1.3, "z": 0.8},
                "center": {"x": 0, "y": 0, "z": -0.2},
            },
        }

    fig.update_layout(
        title=title
        or f"Densidad condicional de {free_label} dado {given_label}",
        template="plotly_white",
        width=1300,
        height=780,
        showlegend=False,
        # The bottom margin has to hold the slider and its tick labels.
        margin={"l": 0, "r": 0, "t": 150, "b": 190},
        scene=scene("f(y | x)", cond.max()),
        scene2=scene("f(x, y)", joint.max()),
        sliders=[
            {
                "active": start,
                "steps": steps,
                "x": 0.1,
                "len": 0.8,
                "y": -0.08,
                "pad": {"t": 30},
                "currentvalue": {
                    "prefix": f"Condicionamos en {given_label} = ",
                    "font": {"size": 15},
                },
            }
        ],
    )
    fig.show()
