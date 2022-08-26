''' Module containing plotting functions.

These plotting functions are used in appConfiguration.py to
generate all the the plots for the UI.
'''


import base64
from io import BytesIO
from PIL import Image
from dash import html

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import scipy.ndimage


IMG_LABEL_SIZE = 12
LAYER_AVG_GAP = 0.8
ATTN_MAP_SCALE_FACTOR = 64

heatmap_layout = dict(
    autosize=True,
    font=dict(color='black'),
    titlefont=dict(color='black', size=14),
    legend=dict(font=dict(size=8), orientation='h'),
)

img_overlay_layout = dict(
    barmode='stack',
    bargap=0,
    hovermode='closest',
    showlegend=False,
    autosize=False,
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(visible=False, fixedrange=True),
    yaxis=dict(visible=False, scaleanchor='x')  # constant aspect ratio
)

def get_sep_indices(tokens):
    return [-1] + [i for i, x in enumerate(tokens) if x == '[SEP]']


def plot_text(db, ex_id):
    tokens = db[ex_id]['tokens']
    txt_len = db[ex_id]['txt_len']
    return show_texts(tokens[:txt_len]), txt_len

def plot_attn_from_txt(db, ex_id, layer, head, word_ids=None, img_coords=None):
    ex_data = db[ex_id]
    txt_len = ex_data['txt_len']
    txt_tokens = ex_data['tokens'][:txt_len]
    if img_coords:
        w, h = db[ex_id]['img_grid_size']
        actual_img_tok_ids = [txt_len + w*(h-j-1) + i for i, j in img_coords]  # TODO double check TODO TODO order image tokens TODO
        token_id = actual_img_tok_ids[0]  # TODO: avg if multiple
    elif word_ids:
        token_id = db.get_txt_token_index(ex_id, word_ids)
    else:  # nothing selected, show empty text
        return show_texts(txt_tokens)
    # get attention
    attn = ex_data['attention']
    if int(head) == head:
        txt_attn = attn[layer, head, token_id, :txt_len]
    else:
        layer_attn = [attn[layer, hd, token_id, :txt_len] for hd in range(attn.shape[0])]
        txt_attn = np.mean(layer_attn, axis=0)
    # get colors by sentence
    colors = cm.get_cmap('Reds')(txt_attn * 120, 0.5)
    colors = ['rgba(' + ','.join(map(lambda x: str(int(x*255)), rgba[:3])) + ',' + str(rgba[3]) + ')'
               for rgba in colors]
    seps = get_sep_indices(ex_data['tokens'])
    colors = [colors[i+1:j+1] for i, j in zip(seps[:-1], seps[1:])]
    txt_fig = show_texts(txt_tokens, colors)
    return txt_fig

def highlight_txt_token(db, ex_id, token_ids):
    txt_len = db[ex_id]['txt_len']
    txt_tokens = db[ex_id]['tokens'][:txt_len]
    if token_ids:
        seps = get_sep_indices(txt_tokens)
        sentence, word = token_ids['sentence'], token_ids['word']
        sentence_len = seps[sentence+1] - seps[sentence]
        colors = []
        for s in range(len(seps) - 1):
            if s == sentence:
                colors.append(['white'] * sentence_len)
                colors[-1][word] = 'orange'
            else:
                colors.append('white')
    else:
        colors = 'white'
    txt_fig = show_texts(txt_tokens, colors)
    return txt_fig

def plot_image(db, ex_id):
    image = db[ex_id]['image']
    return show_img(image, bg='black', opacity=1.0)

def get_empty_fig():
    fig = go.Figure()
    fig.update_layout(img_overlay_layout)
    return fig

def get_overlay_grid(db, ex_id, color='rgba(255, 255, 255, 0)'):
    imgh, imgw, _ = db[ex_id]['image'].shape
    gridw, gridh = db[ex_id]['img_grid_size']
    unit_w = imgw / gridw
    unit_h = imgh / gridh
    grid_data = tuple(go.Bar(
            x=np.linspace(unit_w/2, imgw-unit_w/2, gridw),
            y=[unit_h] * gridw,
            hoverinfo='none',
            marker_line_width=0,
            marker_color=(color if type(color) == str else color[i]))
        for i in range(gridh))
    return grid_data

def get_overlay_fig(db, ex_id, color='rgba(255, 255, 255, 0)'):
    grid_data = get_overlay_grid(db, ex_id, color)
    imgh, imgw, _ = db[ex_id]['image'].shape
    fig = go.Figure(
        data=grid_data,
        layout=go.Layout(
            xaxis={'range': (0, imgw)},
            yaxis={'range': (0, imgh)})
    )
    fig.update_layout(img_overlay_layout)
    return fig

def highlight_img_grid_selection(db, ex_id, img_selection):
    img_selection = set(map(tuple, img_selection))
    w, h = db[ex_id]['img_grid_size']
    highlight, transparent = 'rgba(255, 165, 0, .4)', 'rgba(255, 255, 255, 0)'
    colors = [[(highlight if (j, i) in img_selection else transparent) for j in range(w)] for i in range(h)]
    return get_overlay_fig(db, ex_id, colors)

def plot_attn_from_img(db, ex_id, layer, head, word_ids=None, img_coords=None, smooth=True):
    ex_data = db[ex_id]
    w0, h0 = ex_data['img_grid_size']
    txt_len = ex_data['txt_len']
    if word_ids and word_ids['sentence'] > -1 and word_ids['word'] > -1:
        token_id = db.get_txt_token_index(ex_id, word_ids)
    elif img_coords:
        img_token_ids = [db.get_img_token_index(ex_id, coords) for coords in img_coords]
        token_id = img_token_ids[0]  # TODO show avg for multiple selection?
    else:
        return get_empty_fig()
    attn = ex_data['attention']

    if head < attn.shape[1]:
        img_attn = attn[layer, head, token_id, txt_len:(w0*h0+txt_len)].reshape(h0, w0)
    else:  # show layer avg
        layer_attn = [attn[layer, h, token_id, txt_len:(w0*h0+txt_len)].reshape(h0, w0) \
                      for h in range(attn.shape[0])]
        img_attn = np.mean(layer_attn, axis=0)
    if smooth:
        img_attn = scipy.ndimage.zoom(img_attn, ATTN_MAP_SCALE_FACTOR, order=1)
    return show_img(img_attn, opacity=0.3, bg='rgba(0,0,0,0)', hw=ex_data['image'].shape[:2])

def plot_head_summary(db, ex_id, attn_type_index=0, layer=None, head=None):
    stats = db.get_all_attn_stats(ex_id)
    custom_stats = db.get_custom_metrics(ex_id)
    return show_head_summary(stats, custom_stats, attn_type_index, layer, head)

def plot_attn_matrix(db, ex_id, layer, head, attn_type=0):
    txt_len = db[ex_id]['txt_len']
    txt_attn = db[ex_id]['attention'][layer, head, :txt_len, :txt_len]
    img_attn = db[ex_id]['attention'][layer, head, txt_len:, txt_len:]
    tokens = db[ex_id]['tokens']
    return show_attn_matrix([txt_attn, img_attn], tokens, layer, head, attn_type)


def show_texts(tokens, colors='white'):
    seps = get_sep_indices(tokens)
    sentences = [tokens[i+1:j+1] for i, j in zip(seps[:-1], seps[1:])]

    fig = go.Figure()
    annotations = []
    for sen_i, sentence in enumerate(sentences):
        word_lengths = list(map(len, sentence))
        fig.add_trace(go.Bar(
            x=word_lengths,  # TODO center at 0
            y=[sen_i] * len(sentence),
            orientation='h',
            marker_color=(colors if type(colors) is str else colors[sen_i]),
            marker_line=dict(color='rgba(255, 255, 255, 0)', width=0),
            hoverinfo='none'
        ))
        word_pos = np.cumsum(word_lengths) - np.array(word_lengths) / 2
        for word_i in range(len(sentence)):
            annotations.append(dict(
                xref='x', yref='y',
                x=word_pos[word_i], y=sen_i,
                text=sentence[word_i],
                showarrow=False
            ))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, range=(len(sentences)+.3, -len(sentences)+.7))
    fig.update_layout(
        annotations=annotations,
        barmode='stack',
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig


def show_img(img, opacity, bg, hw=None):
    img_height, img_width = hw if hw else (img.shape[0], img.shape[1])
    mfig, ax = plt.subplots(figsize=(img_width/100., img_height/100.), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.axis('off')
    if hw:
        ax.imshow(img, cmap='jet', interpolation='nearest', aspect='auto')
    else:
        ax.imshow(img)
    img_uri = fig_to_uri(mfig)
    fig_width, fig_height = mfig.get_size_inches() * mfig.dpi

    fig = go.Figure()
    fig.update_xaxes(range=(0, fig_width))
    fig.update_yaxes(range=(0, fig_height))
    fig.update_layout(img_overlay_layout)
    fig.update_layout(
        autosize=True,
        plot_bgcolor=bg,
        paper_bgcolor=bg
    )
    fig.layout.images = []  # remove previous image
    fig.add_layout_image(dict(
        x=0, y=fig_height,
        sizex=fig_width, sizey=fig_height,
        xref='x', yref='y',
        opacity=opacity,
        sizing='stretch',
        source=img_uri
    ))
    return fig


def fig_to_uri(fig, close_all=True, **save_args):
    out_img = BytesIO()
    fig.savefig(out_img, format='jpeg', **save_args)
    if close_all:
        fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode('ascii').replace('\n', '')
    return 'data:image/jpeg;base64,{}'.format(encoded)


def show_head_summary(attn_stats, custom_stats=None, attn_type_index=0, layer=None, head=None):
    n_stats = len(attn_stats) + (len(custom_stats) if custom_stats else 0)
    if attn_type_index >= n_stats:
        attn_type_index = 0

    num_layers, num_heads = np.shape(attn_stats[0])
    data = [go.Heatmap(
        type = 'heatmap',
        x=list(np.arange(-0.5, num_heads-1)) + [num_heads + LAYER_AVG_GAP - 0.5],
        z=stats,
        zmin=-3, zmax=3,  # fix color scale range
        colorscale='plasma',
        reversescale=False,
        colorbar=dict(thickness=10),
        visible=(i == attn_type_index)
    ) for i, stats in enumerate(attn_stats)]

    if custom_stats:
        data += [go.Heatmap(
            type = 'heatmap',
            x=list(np.arange(-0.5, num_heads-1)) + [num_heads + LAYER_AVG_GAP - 0.5],
            z=stats,
            colorscale='plasma',
            reversescale=False,
            colorbar=dict(thickness=10),
            visible=(attn_type_index == len(data))) for stats in custom_stats[1]]

    if layer is not None and head is not None:
        x_pos = np.floor(head) + LAYER_AVG_GAP if head > num_heads - 1 else head
        data.append(dict(
            type='scattergl',
            x=[x_pos],
            y=[layer],
            marker=dict(
                color='black',
                symbol='x',
                size=15,
                opacity=0.6,
                line=dict(width=1, color='lightgrey')
            ),
        ))
    layout = heatmap_layout
    layout.update({'title': ''})
    layout.update({
        'yaxis': {
            'title': 'Layer #',
            'tickmode': 'array',
            'ticktext': list(range(1, num_layers+1)),
            'tickvals': list(range(num_layers)),
            'range': (num_layers-0.5, -0.5),
            'fixedrange': True},
        'xaxis': {
            'title': 'Attention head #',
            'tickmode': 'array',
            'ticktext': list(range(1, num_heads)) + ['Mean'],
            'tickvals': list(range(num_heads-1)) + [num_heads + LAYER_AVG_GAP - 1],
            'range': (-0.5, num_heads + LAYER_AVG_GAP - 0.5),
            'fixedrange': True},
        'margin': {'t': 0, 'l': 0, 'r': 0, 'b': 0},
        'height': 450,
        'width': 520
    })

    fig = go.Figure(data=data, layout=layout)
    fig.add_vrect(x0=num_heads-1.5, x1=num_heads+LAYER_AVG_GAP-1.5, fillcolor='white', line_width=0)

    # dropdown menu
    labels = ['Mean image-to-text attention (without CLS/SEP)',
              'Mean text-to-image attention (without CLS/SEP)',
              'Mean image-to-image attention (without CLS/SEP)',
              'Mean image-to-image attention (without self/CLS/SEP)',
              'Mean text-to-text attention (without CLS/SEP)',
              'Mean cross-modal attention (without CLS/SEP)',
              'Mean intra-modal attention (without CLS/SEP)']
    if custom_stats:
        labels += custom_stats[0]
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(
                    args=[{'visible': [True if j == i else False for j in range(len(labels))] + [True]}],
                    label=labels[i],
                    method='update'
                ) for i in range(len(labels))
            ],
            direction='down',
            pad={'l': 85},
            active=attn_type_index,
            showactive=True,
            x=0,
            xanchor='left',
            y=1.01,
            yanchor='bottom'
        )],
        annotations=[dict(
            text='Display data:', showarrow=False,
            x=0, y=1.027, xref='paper', yref='paper',
            yanchor='bottom'
        )]
    )

    return fig


def show_attn_matrix(attns, tokens, layer, head, attn_type):
    txt_tokens = tokens[:attns[0].shape[0]]
    data = [dict(
        type='heatmap',
        z=attn,
        colorbar=dict(thickness=10),
        visible=(i == attn_type)) for i, attn in enumerate(attns)]
    layout = heatmap_layout
    layout.update({
        'title': f'Attention Matrix for Layer #{layer+1}, Head #{head+1}',
        'title_font_size': 16,
        'title_x': 0.5, 'title_y': 0.99,
        'margin': {'t': 70, 'b': 10, 'l': 0, 'r': 0}
    })
    txt_attn_layout = {
        'xaxis': dict(
            tickmode='array',
            tickvals=list(range(len(txt_tokens))),
            ticktext=txt_tokens,
            tickangle=45,
            tickfont=dict(size=12),
            range=(-0.5, len(txt_tokens)-0.5),
            fixedrange=True),
        'yaxis': dict(
            tickmode='array',
            tickvals=list(range(len(txt_tokens))),
            ticktext=txt_tokens,
            tickfont=dict(size=12),
            range=(-0.5, len(txt_tokens)-0.5),
            fixedrange=True),
        'width': 600,
        'height': 600
    }
    img_attn_layout = {
        'xaxis': dict(range=(-0.5, len(tokens)-len(txt_tokens)-0.5)),
        'yaxis': dict(range=(-0.5, len(tokens)-len(txt_tokens)-0.5), scaleanchor='x'),
        'width': 900,
        'height': 900,
        'plot_bgcolor': 'rgba(0,0,0,0)',
    }
    layout.update((txt_attn_layout, img_attn_layout)[attn_type])
    figure = go.Figure(dict(data=data, layout=layout))

    # dropdown menu
    figure.update_layout(
        updatemenus=[dict(
            buttons=[dict(
                args=[
                    {'visible': [True, False]},
                    txt_attn_layout
                ],
                label='Text-to-text attention',
                method='update'
            ), dict(
                args=[
                    {'visible': [False, True]},
                    img_attn_layout
                ],
                label='Image-to-image attention',
                method='update'
            )],
            direction='down',
            pad={'l': 85},
            active=attn_type,
            showactive=True,
            x=0,
            xanchor='left',
            y=1.01,
            yanchor='bottom'
        )],
        annotations=[dict(
            text='Display data:', showarrow=False,
            x=0, y=1.02,
            xref='paper', yref='paper',
            xanchor='left', yanchor='bottom'
        )],
    )

    return figure

def show_tsne(db, ex_id, layer, text_tokens_id, img_tokens):
    ex_data = db[ex_id]
    selected_token_id = None
    if text_tokens_id and text_tokens_id['sentence'] >= 0 and text_tokens_id['word'] >= 0:
        selected_token_id = db.get_txt_token_index(ex_id, text_tokens_id)
        mod = 'img'
    elif img_tokens:
        selected_token_id = db.get_img_token_index(ex_id, img_tokens[0])
        mod = 'txt'

    # draw all tokens
    figure = go.Figure()
    txt_len = ex_data['txt_len']
    for modality in ('txt', 'img'):
        slicing = slice(txt_len) if modality == 'txt' else slice(txt_len, len(ex_data['tokens']))
        figure.add_trace(go.Scatter(
            x=ex_data['tsne'][layer][slicing][:,0],
            y=ex_data['tsne'][layer][slicing][:,1],
            hovertext=[f'{ex_id},{token_id}' for token_id in list(range(len(ex_data['tokens'])))[slicing]],
            mode='markers',
            name=modality,
            marker={'color': 'blue' if modality == 'img' else 'red'}
        ))
    if selected_token_id is not None:
        # hightlight selected
        token = ex_data['tokens'][selected_token_id]
        tsne = ex_data['tsne'][layer][selected_token_id]
        figure.add_trace(go.Scatter(
            x=[tsne[0]],
            y=[tsne[1]],
            mode='markers+text',
            hovertext=[f'{ex_id},{selected_token_id}'],
            text=token,
            textposition='top center',
            textfont=dict(size=15),
            marker=dict(
                color='orange',
                colorscale='Jet',
                showscale=False,
                symbol='star',
                size=10,
                opacity=1,
                cmin=-1.0,
                cmax=1.0,
            ),
            showlegend=False,
        ))
        # hightlight closest
        closest_ex_id, closest_token_id = db.find_closest_token(tsne, layer, mod)
        closest_token = db[closest_ex_id]['tokens'][closest_token_id]
        closest_tsne = db[closest_ex_id]['tsne'][layer][closest_token_id]
        figure.add_trace(go.Scatter(
            x=[closest_tsne[0]],
            y=[closest_tsne[1]],
            mode='markers+text',
            hovertext=[f'{closest_ex_id},{closest_token_id}'],
            text=closest_token + ' from ex ' + str(closest_ex_id),
            textposition='top center',
            textfont=dict(size=15),
            marker=dict(
                color='green',
                colorscale='Jet',
                showscale=False,
                symbol='star',
                size=10,
                opacity=1,
                cmin=-1.0,
                cmax=1.0,
            ),
            showlegend=False,
        ))
    figure.update_layout(legend=dict(font=dict(size=15), y=1, x=0.99, bgcolor='rgba(0,0,0,0)'))
    figure.update_layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    figure.update_traces(hoverinfo='none', hovertemplate=None)
    return figure


def plot_tooltip_content(db, ex_id, token_id):
    ex_data = db[ex_id]
    txt_len = ex_data['txt_len']
    token = ex_data['tokens'][token_id]
    if token_id < txt_len:  # text token
        return [html.P(token)]
    # image token
    img_coords = ex_data['img_coords'][token_id-txt_len]
    img = np.copy(ex_data['image'])
    x_unit_len, y_unit_len = db.get_img_unit_len(ex_id)
    x = int(x_unit_len * img_coords[0])
    y_end = img.shape[0] - int(y_unit_len * img_coords[1])
    x_end = x + x_unit_len
    y = y_end - y_unit_len
    img[y:y_end, x:x_end, 0] = 0
    img[y:y_end, x:x_end, 1] = 0
    img[y:y_end, x:x_end, 2] = 255
    # dump img to base64
    buffer = BytesIO()
    img = Image.fromarray(np.uint8(img)).save(buffer, format='jpeg')
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    img_url = 'data:image/jpeg;base64, ' + encoded_image
    return [
        html.Div([
            html.Img(
                src=img_url,
                style={'width': '300px', 'display': 'block', 'margin': '0 auto'},
            ),
            html.P(token, style={'font-weight': 'bold','font-size': '7x'})
        ])
    ]
