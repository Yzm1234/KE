import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
          "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
          "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
          "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
          "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
          "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
          "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
          "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
          "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
          "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
          "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
          "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
          "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
          "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
          "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


def interactive_plot(x, y, labels, title='My Plot', save=False, plot_name=None):
    """
    This method is deprecated, because it is using graphic object in plotly, please see plot_pca/plot_tsne instead
    :param x:
    :type x:
    :param y:
    :type y:
    :param labels:
    :type labels:
    :param title:
    :type title:
    :param save:
    :type save:
    :param plot_name:
    :type plot_name:
    :return:
    :rtype:
    """
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
                      paper_bgcolor="LightSteelBlue", )

    fig.show()
    if save:
        fig.write_html(plot_name)


def plot_pca(df, first_feature_numerical_col_idx=5, save=False, saved_file_name='2D_PCA_colored_by_biome.html'):
    X = df.iloc[:, first_feature_numerical_col_idx:].to_numpy()
    pca_2d = PCA(n_components=2).fit_transform(X)
    df['pca_2d_x1'] = pca_2d[:, 0]
    df['pca_2d_x2'] = pca_2d[:, 1]
    fig = px.scatter(df,
                     x='pca_2d_x1',
                     y='pca_2d_x2',
                     color='biome',
                     color_discrete_sequence=colors,
                     hover_name='biome',
                     hover_data={'pca_2d_x1': False,
                                 'pca_2d_x2': False,
                                 'biome': False,
                                 'id': True,
                                 'study_id': True,
                                 'sample_id': True,
                                 'exptype': True
                                 },
                     title=os.path.splitext(saved_file_name)[0])

    fig.update_traces(marker=dict(size=13,
                                  line=dict(width=1.5,
                                            color='MediumPurple'),
                                  opacity=0.8),
                      selector=dict(mode='markers'))

    fig.update_layout(autosize=False,
                      width=1750,
                      height=950,
                      paper_bgcolor="LightSteelBlue", )
    fig.show()
    if save:
        fig.write_html(saved_file_name)


def plot_tsne(df, first_feature_numerical_col_idx=5, save=False, saved_file_name='2D_TSNE_colored_by_biome.html'):
    X = df.iloc[:, first_feature_numerical_col_idx:].to_numpy()
    tsne_2d = TSNE(n_components=2).fit_transform(X)
    df['tsne_2d_x1'] = tsne_2d[:, 0]
    df['tsne_2d_x2'] = tsne_2d[:, 1]
    fig = px.scatter(df,
                     x='tsne_2d_x1',
                     y='tsne_2d_x2',
                     color='biome',
                     color_discrete_sequence=colors,
                     hover_name='biome',
                     hover_data={'tsne_2d_x1': False,
                                 'tsne_2d_x2': False,
                                 'biome': False,
                                 'id': True,
                                 'study_id': True,
                                 'sample_id': True,
                                 'exptype': True
                                 },
                     title=os.path.splitext(saved_file_name)[0])

    fig.update_traces(marker=dict(size=13,
                                  line=dict(width=1.5,
                                            color='MediumPurple'),
                                  opacity=0.8),
                      selector=dict(mode='markers'))

    fig.update_layout(autosize=False,
                      width=1750,
                      height=950,
                      paper_bgcolor="LightSteelBlue", )
    fig.show()
    if save:
        fig.write_html(saved_file_name)


# def plot_pyplot():
#     fig, ax = plt.subplots()
#     ax.axis('off')
#     ax.format_coord = lambda x, y: ''
#     # fig.set_size_inches(8, 6)
#     fig.subplots_adjust(right=0.7, top=0.7)
#     cmap = plt.cm.get_cmap('tab20')
#     sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=65, c=np.random.rand(94), cmap=cmap)
#
#     annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), fontsize=10, textcoords="offset points",  # 'axes fraction',
#                         bbox=dict(boxstyle="round", fc="w"),
#                         arrowprops=dict(arrowstyle="->"), annotation_clip=False)
#
#     def update_annot(ind):
#         pos = sc.get_offsets()[ind["ind"][0]]
#         annot.xy = pos
#         text = "{}".format(" ".join([y[n] for n in ind["ind"]]))
#         annot.set_text(text)
#         annot.get_bbox_patch().set_alpha(0.4)
#
#     def hover(event):
#         vis = annot.get_visible()
#         if event.inaxes == ax:
#             cont, ind = sc.contains(event)
#             if cont:
#                 update_annot(ind)
#                 annot.set_visible(True)
#                 fig.canvas.draw_idle()
#             else:
#                 if vis:
#                     annot.set_visible(False)
#                     fig.canvas.draw_idle()
#
#     fig.canvas.mpl_connect("motion_notify_event", hover)
#     fig.suptitle('Biome PCA', fontsize=16)
#     plt.show()

