import base64
import html

import folium
from folium.plugins import MarkerCluster
from IPython.display import HTML, IFrame

#: Marker colours handed out to labels, in sorted-label order.
PALETTE = [
    "blue",
    "red",
    "green",
    "purple",
    "orange",
    "darkred",
    "cadetblue",
    "darkgreen",
    "pink",
    "gray",
]

#: Group name for rows whose label is missing.
NO_LABEL = "Sin etiqueta"

#: CSS colour of each ``folium.Icon`` colour, for the legend dots. Several
#: marker names ("lightred", "darkpurple") are not CSS colours.
MARKER_HEX = {
    "red": "#d63e2a",
    "darkred": "#a23336",
    "lightred": "#ff8e7f",
    "orange": "#f69730",
    "beige": "#ffcb92",
    "green": "#72b026",
    "darkgreen": "#728224",
    "lightgreen": "#bbf970",
    "blue": "#38aadd",
    "darkblue": "#00649f",
    "lightblue": "#8adaff",
    "purple": "#d252b9",
    "darkpurple": "#5b396b",
    "pink": "#ff91ea",
    "cadetblue": "#436978",
    "white": "#fbfbfb",
    "gray": "#575757",
    "lightgray": "#a3a3a3",
    "black": "#303030",
}


def create_map(
    df,
    latitude_col="latitude",
    longitude_col="longitude",
    name_col=None,
    label_col=None,
    colors=None,
    cluster=True,
):
    """
    Create a folium map with one marker per row of the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing latitude and longitude columns.
    latitude_col (str): Name of the latitude column in the DataFrame.
    longitude_col (str): Name of the longitude column in the DataFrame.
    name_col (str or None): Column with each point's name, shown on hover
        and at the top of its popup.
    label_col (str or None): Column whose values colour the markers. Each
        label becomes a layer that the map's legend can switch on and off;
        missing labels are grouped under "Sin etiqueta".
    colors (dict or None): ``{label: colour}`` overriding the default
        palette, using ``folium.Icon`` colour names.
    cluster (bool): Group nearby markers into count bubbles. The bubbles
        hide the label colours, so turn it off for a few hundred points.

    Returns:
    folium.Map: A folium map object with the markers.
    """
    clean_df = df.dropna(subset=[latitude_col, longitude_col])
    if clean_df.empty:
        raise ValueError(
            """The DataFrame is empty after dropping rows with NaN values in
            latitude and longitude columns."""
        )

    # create the map
    mapa = folium.Map(
        location=[
            clean_df[latitude_col].mean(),
            clean_df[longitude_col].mean(),
        ],
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    # one group per label, or a single group holding every point
    if label_col is None:
        groups = {None: clean_df}
        palette = {None: None}
    else:
        labels = clean_df[label_col].astype(object).fillna(NO_LABEL)
        groups = dict(tuple(clean_df.groupby(labels, sort=True)))
        palette = {
            label: PALETTE[i % len(PALETTE)]
            for i, label in enumerate(k for k in groups if k != NO_LABEL)
        }
        palette[NO_LABEL] = "gray"
        palette.update(colors or {})
        unknown = {
            label: color
            for label, color in palette.items()
            if label in groups and color not in folium.Icon.color_options
        }
        if unknown:
            raise ValueError(
                f"Not folium.Icon colours: {unknown}. "
                f"Use one of {sorted(folium.Icon.color_options)}."
            )

    for label, rows in groups.items():
        if label is None:
            layer = mapa
        else:
            dot = MARKER_HEX[palette[label]]
            layer = folium.FeatureGroup(
                name=(
                    f'<span style="color:{dot}">&#9679;</span> '
                    f"{html.escape(str(label))}"
                )
            ).add_to(mapa)
        # add the points to a cluster instead of directly to the map
        target = MarkerCluster().add_to(layer) if cluster else layer

        for _, row in rows.iterrows():
            lat, lon = row[latitude_col], row[longitude_col]
            popup = f"Lat: {lat}, Lon: {lon}"
            tooltip = None
            if name_col is not None:
                tooltip = html.escape(str(row[name_col]))
                popup = f"<b>{tooltip}</b><br>{popup}"
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=tooltip,
                icon=(
                    None
                    if palette[label] is None
                    else folium.Icon(color=palette[label])
                ),
            ).add_to(target)

    if label_col is not None:
        folium.LayerControl(collapsed=False).add_to(mapa)

    return mapa


def show_map(mapa, height=500, inline=False):
    """
    Display a folium map in a notebook cell, VS Code included.

    folium's own display puts the page in an ``<iframe srcdoc=...>``, which
    VS Code's output view leaves blank behind the "Make this Notebook
    Trusted" text. This passes the page as a base64 ``data:`` URI instead,
    as older folium versions did.

    Parameters:
    mapa (folium.Map): The map to display, e.g. from ``create_map``.
    height (int): Height of the map in pixels.
    inline (bool): Embed the page directly, without an iframe. A fallback
        for viewers that also block ``data:`` iframes; folium's CSS then
        applies to the whole output.

    Returns:
    IPython.display.IFrame or IPython.display.HTML: The cell's output.
    """
    page = mapa.get_root().render()
    if inline:
        return HTML(f'<div style="height:{height}px;">{page}</div>')
    encoded = base64.b64encode(page.encode("utf-8")).decode("ascii")
    return IFrame(
        src=f"data:text/html;charset=utf-8;base64,{encoded}",
        width="100%",
        height=height,
    )
