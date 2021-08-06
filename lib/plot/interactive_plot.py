import numpy as np
import plotly.graph_objects as go


def interactive_plot(x, y, labels, title='My Plot', save=False, plot_name=None):
    fig = go.Figure(data=go.Scatter(x=x,
                                    y=y,
                                    mode='markers',
                                    text=labels,
                                    hovertemplate="%{text}<extra></extra>",
                                    showlegend=False,
                                    marker=dict(
                                        size=15,
                                        color=np.random.randn(94),
                                        colorscale='Viridis'
                                    )))

    fig.update_layout(title=title,
                      autosize=False,
                      width=1000,
                      height=1000,
                      paper_bgcolor="LightSteelBlue",)

    fig.show()
    if save:
        fig.write_html(plot_name)


def pca_plot():
    pass


def tsne_plot():
    pass

