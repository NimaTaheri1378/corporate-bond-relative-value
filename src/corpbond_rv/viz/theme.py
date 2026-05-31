from __future__ import annotations


def apply_plotly_template() -> None:
    """Register a clean institutional Plotly template."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return

    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, Arial, sans-serif", size=13),
            title=dict(x=0.02, xanchor="left", font=dict(size=22)),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=70, r=35, t=80, b=60),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(30, 41, 59, 0.10)",
                zeroline=False,
                linecolor="rgba(30, 41, 59, 0.35)",
                ticks="outside",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(30, 41, 59, 0.10)",
                zeroline=False,
                linecolor="rgba(30, 41, 59, 0.35)",
                ticks="outside",
            ),
        )
    )
    pio.templates["corp_bond_rv"] = template
    pio.templates.default = "corp_bond_rv"
