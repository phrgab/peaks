import os

import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.transform import dodge, factor_cmap


def ptab(modify_elements_func=None, add_to_tooltips=None):
    """Plot the periodic table
    Modified from https://docs.bokeh.org/en/2.4.3/docs/gallery/periodic.html

    Parameters
    ----------
    modify_elements_func : callable, optional
        Function to modify the elements data frame. Should accept a single argument of
        the elements data frame and return a modified data frame.
    add_to_tooltips : tuple | list of tuples, optional
        Additional tooltips to add to the plot.
    """
    output_notebook()

    # Load the elements data
    # elements.pkl is a DataFrame bundled from bokeh.sample_data, License: Public Domain
    elements_path = os.path.join(os.path.dirname(__file__), "elements.pkl")
    elements = pd.read_pickle(elements_path)

    # Define period and group information
    periods = ["I", "II", "III", "IV", "V", "VI", "VII", "", "*", "**"]
    groups = [str(x) for x in range(1, 19)]

    # Create Data frame for main group elements
    global df
    df = elements.copy()
    df["atomic mass"] = df["atomic mass"].astype(str)
    df["period"] = [periods[x - 1] for x in df.period]

    # Modify group and period information of LA and AC series for plotting purposes
    for ii in range(56, 71):
        df.loc[ii, "period"] = "*"
        df.loc[ii, "group"] = 4 + ii - 56
    for ii in range(88, 103):
        df.loc[ii, "period"] = "**"
        df.loc[ii, "group"] = 4 + ii - 88
    df["group"] = df["group"].astype(str)
    df.loc[70, "metal"] = "lanthanoid"
    df.loc[102, "metal"] = "actinoid"

    # Correct name for Hn
    df.loc[112, "name"] = "Nihonium"

    # Modify the elements data frame if necessary
    if modify_elements_func:
        df = modify_elements_func(df)

    cmap = {
        "alkali metal": "#a6cee3",
        "alkaline earth metal": "#1f78b4",
        "metal": "#d93b43",
        "halogen": "#999d9a",
        "metalloid": "#e08d49",
        "noble gas": "#eaeaea",
        "nonmetal": "#f1d4Af",
        "transition metal": "#65C99C",
        "lanthanoid": "#FFFFE0",
        "actinoid": "#FFD9D9",
    }

    TOOLTIPS = [
        ("Name", "@name"),
        ("Atomic number", "@{atomic number}"),
        ("Atomic mass", "@{atomic mass}"),
        ("Type", "@metal"),
        ("Electronic configuration", "@{electronic configuration}"),
    ]
    # Add additional tooltips if necessary
    if add_to_tooltips:
        if isinstance(add_to_tooltips, tuple):
            TOOLTIPS.append(add_to_tooltips)
        else:
            TOOLTIPS.extend(add_to_tooltips)

    p = figure(
        title="Periodic Table",
        width=1000,
        height=550,
        x_range=groups,
        y_range=list(reversed(periods)),
        tools="hover",
        toolbar_location=None,
        tooltips=TOOLTIPS,
    )

    r = p.rect(
        "group",
        "period",
        0.95,
        0.95,
        source=df,
        fill_alpha=0.6,
        legend_field="metal",
        color=factor_cmap(
            "metal", palette=list(cmap.values()), factors=list(cmap.keys())
        ),
    )

    text_props = dict(source=df, text_align="left", text_baseline="middle")

    x = dodge("group", -0.4, range=p.x_range)

    p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

    p.text(
        x=x,
        y=dodge("period", 0.3, range=p.y_range),
        text="atomic number",
        text_font_size="11px",
        **text_props,
    )

    p.text(
        x=x,
        y=dodge("period", -0.35, range=p.y_range),
        text="name",
        text_font_size="7px",
        **text_props,
    )

    p.text(
        x=x,
        y=dodge("period", -0.2, range=p.y_range),
        text="atomic mass",
        text_font_size="7px",
        **text_props,
    )

    p.text(
        x=["3", "3"],
        y=["VI", "VII"],
        text=["LA", "AC"],
        text_align="center",
        text_baseline="middle",
    )

    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = "horizontal"
    p.legend.location = (0, 105)
    p.hover.renderers = [r]

    show(p)
