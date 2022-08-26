''' Module containing the main Dash app code.

The main function in this module is configureApp() which starts 
the Dash app on a Flask server and configures it with callbacks.
'''

import os
import copy
import dash
from flask import Flask
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from termcolor import cprint
from .app_layout import get_layout
from .plot_func import *
from app.database.db_analyzer import VliDataBaseAnalyzer


def configure_app(db_dir, model=None):
    db = VliDataBaseAnalyzer(db_dir, read_only=(not model))
    n_layers = db['n_layers']
    app = start_app(db['n_examples'], n_layers)

    @app.callback(
        Output('add_ex_div', 'style'),
        Output('add_ex_toggle', 'style'),
        Input('add_ex_toggle', 'n_clicks'),
    )
    def toggle_add_example(n_clicks):
        if model is None:
            return {'display': 'none'}, {'display': 'none'}
        if not n_clicks:
            raise PreventUpdate
        if n_clicks % 2 == 0:
            return {'display': 'none'}, {'padding': '0 15px'}
        return {}, {'padding': '0 15px', 'background': 'lightgrey'}


    @app.callback(
        Output('ex_selector', 'max'),
        Output('ex_selector', 'value'),
        Input('add_ex_btn', 'n_clicks'),
        State('new_ex_img', 'value'),
        State('new_ex_txt', 'value'),
    )
    def add_example(n_clicks, image_in, text_in):
        if not n_clicks:  # app initialzation
            return db['n_examples'] - 1, '0'
        ex_id = db['n_examples']
        data = model.data_setup(ex_id, image_in, text_in)
        db.add_example(ex_id, data)
        return ex_id, ex_id


    @app.callback(
        Output('ex_text', 'figure'),
        Output('ex_txt_len', 'data'),
        Output('ex_sm_text', 'figure'),
        Output('ex_img_attn_overlay', 'figure'),
        [Input('ex_selector', 'value'),
         Input('selected_head_layer', 'data'),
         Input('selected_text_token', 'data'),
         Input('selected_img_tokens', 'data'),
         Input('map_smooth_checkbox', 'value')],
        [State('ex_txt_len', 'data')]
    )
    def display_ex_text_and_img_overlay(ex_id, head_layer, text_token, selected_img_token, smooth,
                                        txt_len):
        trigger = get_input_trigger(dash.callback_context)
        layer, head = head_layer['layer'], head_layer['head']
        new_state = (layer < 0 or head < 0) or (not txt_len)
        if (not selected_img_token) and \
                (not text_token or text_token['sentence'] < 0 or text_token['word'] < 0):
            new_state = True

        if new_state or trigger == 'ex_selector':
            text_fig, txt_len = plot_text(db, ex_id)
            text_attn_fig = text_fig
            img_attn_overlay = get_empty_fig()
            return text_fig, txt_len, text_attn_fig, img_attn_overlay

        img_token = set(map(tuple, selected_img_token))
        text_fig = highlight_txt_token(db, ex_id, text_token)
        if trigger == 'selected_text_token':
            img_token = None
        elif trigger == 'selected_img_tokens':
            text_token = None
            text_fig, txt_len = plot_text(db, ex_id)
        
        text_attn_fig = plot_attn_from_txt(db, ex_id, layer, head, text_token, img_token)
        img_attn_overlay = plot_attn_from_img(db, ex_id, layer, head, word_ids=text_token,
                                              img_coords=img_token, smooth=smooth)

        return text_fig, txt_len, text_attn_fig, img_attn_overlay


    @app.callback(
        Output('ex_img_token_overlay', 'figure'),
        [Input('ex_selector', 'value'),
         Input('selected_img_tokens', 'data')]
    )
    def display_img_token_selection(ex_id, selected_img_tokens):
        if not ex_id:
            return get_empty_fig()
        trigger = get_input_trigger(dash.callback_context)
        if trigger == 'ex_selector':
            token_overlay = get_overlay_fig(db, ex_id)
        else:
            token_overlay = highlight_img_grid_selection(db, ex_id, selected_img_tokens)
        return token_overlay


    @app.callback(
        Output('attn2img', 'style'),
        Output('attn2txt', 'style'),
        Output('attn2img_label', 'style'),
        Output('attn2txt_label', 'style'),
        Output('play_btn', 'style'),
        [Input('attn2img_toggle', 'value')]
    )
    def toggle_attn_to(attn_toggle):
        hidden_style = {'display': 'none', 'z-index': '-1'}
        if attn_toggle:  # hide image
            return hidden_style, {}, {'color': 'lightgrey'}, {}, {}
        else:  # hide text
            return {}, hidden_style, {}, {'color': 'lightgrey'}, hidden_style


    @app.callback(
        Output('ex_image', 'figure'),
        Output('ex_sm_image', 'figure'),
        [Input('ex_selector', 'value')]
    )
    def display_ex_image(ex_id):
        image_fig = plot_image(db, ex_id)
        sm_image_fig = copy.deepcopy(image_fig)
        sm_image_fig.update_layout(height=256)
        return image_fig, sm_image_fig


    @app.callback(
        Output('selected_img_tokens', 'data'),
        [Input('ex_img_token_overlay', 'clickData'),
         Input('ex_img_token_overlay', 'selectedData'),
         Input('selected_text_token', 'data'),
         Input('attn2img_toggle', 'value')],  # clear if toggle attention to
        [State('selected_img_tokens', 'data')]
    )
    def save_img_selection(click_data, selected_data, txt_token, toggle, points):
        if points is None:
            return []
        if (not click_data) and (not selected_data):
            return []
        trigger = get_input_trigger_full(dash.callback_context).split('.')[1]
        points = set(map(tuple, points))
        if trigger == 'clickData':
            x = click_data['points'][0]['pointNumber']
            y = click_data['points'][0]['curveNumber']
            if (x, y) in points:
                points.remove((x, y))
            else:
                # points.add((x, y))
                points = {(x, y)}  # TODO: enable a set of multiple points and show average
        elif trigger == 'selectedData':
            if selected_data:
                points.update([(pt['pointNumber'], pt['curveNumber']) for pt in selected_data['points']])
            else:
                return []
        else:
            return []
        return list(points)  # list is JSON serializable


    @app.callback(
        Output('selected_text_token', 'data'),
        [Input('ex_text', 'clickData'),       # from mouse click
         Input('movie_progress', 'data'),     # from movie slider update
         Input('ex_selector', 'value'),       # clear if new example selected
         Input('attn2img_toggle', 'value')],  # clear if toggle attention to
        [State('auto_stepper', 'disabled')]
    )
    def save_text_click(click_data, movie_progress, ex_id, toggle, movie_disabled):
        trigger = get_input_trigger(dash.callback_context)
        if trigger == 'ex_text' and click_data:
            sentence_id = click_data['points'][0]['curveNumber']
            word_id = click_data['points'][0]['pointIndex']
            return {'sentence': sentence_id, 'word': word_id}
        if trigger == 'movie_progress':
            if movie_disabled or movie_progress == -1:  # no change
                raise PreventUpdate()
            # convert index to sentence/word
            sentence, word = 0, 0
            txt_len = db[ex_id]['txt_len']
            for i, token in enumerate(db[ex_id]['tokens'][:txt_len]):
                if i == movie_progress:
                    break
                if token == '[SEP]':
                    sentence += 1
                    word = -1
                word += 1
            return {'sentence': sentence, 'word': word}
        return None


    @app.callback(
        Output('head_summary', 'figure'),
        [Input('ex_selector', 'value'),
         Input('selected_head_layer', 'data')],
        [State('head_summary', 'figure')]
    )
    def display_head_summary(ex_id, head_layer, fig):
        layer, head = head_layer['layer'], head_layer['head']
        attn_type_index = get_active_figure_data(fig)
        data = plot_head_summary(db, ex_id, attn_type_index, layer, head)
        return data


    @app.callback(
        Output('selected_head_layer', 'data'),
        [Input('head_summary', 'clickData')]
    )
    def save_head_summary_click(click_data):
        if click_data:
            x, y = get_click_coords(click_data)
            return {'layer': y, 'head': x}
        return {'layer': 10, 'head': 7}


    @app.callback(
        Output('tsne_title', 'children'),
        Output('attn_title', 'children'),
        Output('accuracy', 'children'),
        [Input('layer_slider', 'value'),
         Input('selected_head_layer', 'data'),
         Input('ex_selector', 'value')]
    )
    def update_titles(tsne_layer, selected_head_layer, ex_id):
        if ('accuracy' not in db[ex_id]) or (db[ex_id]['accuracy'] is None):
            acc_text = 'unknown'
        else:
            acc_text = 'correct' if db[ex_id]['accuracy'] else 'incorrect'
        if tsne_layer == n_layers:
            tsne_title = 't-SNE Embeddings before Layer #1'
        else:
            tsne_title = f't-SNE Embeddings after Layer #{n_layers - tsne_layer}'
        layer, head = selected_head_layer['layer'], selected_head_layer['head']
        head = 'Average' if int(head) != head else f'#{head + 1}'
        att_title = f'Attention in Layer #{layer+1}, Head {head}'
        return tsne_title, att_title, acc_text


    # movie callbacks

    @app.callback(
        Output('movie_progress', 'data'),
        [Input('auto_stepper', 'n_intervals'),
         Input('auto_stepper', 'disabled'),
         Input('ex_selector', 'value')],
        [State('movie_progress', 'data'),
         State('ex_txt_len', 'data')]
    )
    def stepper_advance(n_intervals, disable, ex_id, movie_progress, txt_len):
        trigger = get_input_trigger_full(dash.callback_context)
        if trigger.startswith('ex_selector') or \
                (trigger == 'auto_stepper.disabled' and disable):
            return -1
        if n_intervals:
            return (movie_progress + 1) % txt_len
        return -1

    @app.callback(
        Output('play_btn', 'children'),
        Output('play_btn', 'n_clicks'),
        Output('auto_stepper', 'disabled'),
        [Input('play_btn', 'n_clicks'),
         Input('ex_selector', 'value'),
         Input('attn2img_toggle', 'value')],
        [State('auto_stepper', 'n_intervals')]
    )
    def play_btn_click(n_clicks, ex_id, attn_to_toggle, n_intervals):
        trigger = get_input_trigger(dash.callback_context)
        if (trigger == 'play_btn') and (n_intervals is not None) and \
                (n_clicks is not None) and n_clicks % 2 == 1:
            return '\u275A\u275A', n_clicks, False
        elif (trigger == 'ex_selector') or (trigger == 'attn2img_toggle'):
            return '\u25B6', 0, True
        return '\u25B6', n_clicks, True


    # TSNE

    @app.callback(
        Output('tsne_map', 'figure'),
        [Input('ex_selector', 'value'),
         Input('layer_slider', 'value'),
         Input('selected_text_token', 'data'),
         Input('selected_img_tokens', 'data')],
        [State('auto_stepper', 'disabled')]
    )
    def plot_tsne(ex_id, layer, text_tokens_id, img_tokens, movie_paused):
        if not movie_paused:
            raise PreventUpdate
        figure = show_tsne(db, ex_id, n_layers - layer, text_tokens_id, img_tokens)
        return figure

    @app.callback(
        Output('hover_out', 'show'),
        Output('hover_out', 'bbox'),
        Output('hover_out', 'children'),
        Input('tsne_map', 'hoverData'),
        Input('layer_slider', 'value')
    )
    def display_hover(hover_data, layer):
        if hover_data is None:
            return False, dash.no_update, dash.no_update
        layer = n_layers - layer
        point = hover_data['points'][0]
        ex_id, token_id = point['hovertext'].split(',')
        tooltip_children = plot_tooltip_content(db, ex_id, int(token_id))
        return True, point['bbox'], tooltip_children


    return app


def start_app(n_examples, n_layers):
    print('Starting server')
    server = Flask(__name__)
    server.secret_key = os.environ.get('secret_key', 'secret')

    app = dash.Dash(__name__, server=server, url_base_pathname='/')
    app.title = 'VL-InterpreT'
    app.layout = get_layout(n_examples, n_layers)
    return app

def print_page_link(hostname, port):
    print('\n\n')
    cprint('------------------------' '------------------------', 'green')
    cprint('App Launched!', 'red')
    cprint('------------------------' '------------------------', 'green')

def get_click_coords(click_data):
    return click_data['points'][0]['x'], click_data['points'][0]['y']

def get_input_trigger(ctx):
    return ctx.triggered[0]['prop_id'].split('.')[0]

def get_input_trigger_full(ctx):
    return ctx.triggered[0]['prop_id']

def get_active_figure_data(fig):
    return fig['layout']['updatemenus'][0]['active'] if fig else 0
