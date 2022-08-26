""" Module specifying Dash app UI layout

The main function here is the get_layout() function which returns
the Dash/HTML layout for InterpreT.
"""

from dash import dcc
from dash import html
import dash_daq as daq
import base64
import os

intel_dark_blue = "#0168b5"
intel_light_blue = "#04c7fd"


def get_layout(n_examples, n_layers):
    logoConfig = dict(displaylogo=False, modeBarButtonsToRemove=["sendDataToCloud"])
    image_filename = os.path.join("app", "assets", "intel_ai_logo.jpg")
    encoded_image = base64.b64encode(open(image_filename, "rb").read())

    layout = html.Div(
        [
            # Stored values
            dcc.Store(id="selected_head_layer"),
            dcc.Store(id="selected_text_token", data={'sentence': -1, 'word': -1}),
            dcc.Store(id="selected_img_tokens"),
            dcc.Store(id="selected_token_from_matrix"),
            dcc.Store(id="movie_progress", data=0),
            dcc.Store(id="ex_txt_len"),


            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(  # Intel logo
                                src=f"data:image/png;base64,{encoded_image.decode()}",
                                style={
                                    "display": "inline",
                                    "height": str(174 * 0.18) + "px",
                                    "width": str(600 * 0.18) + "px",
                                    "position": "relative",
                                    "padding-right": "30px",
                                    "vertical-align": "middle",
                                },
                            ),
                            html.Div([
                                    html.Span("VL-Interpre", style={'color': intel_dark_blue}),
                                    html.Span("T", style={'color': intel_light_blue})
                                ], style={"font-size": "40px", "display": "inline", "vertical-align": "middle"}
                            ),
                            html.Div(
                                children=[
                                    html.Span("An Interactive Visualization Tool for "),
                                    html.Strong("Interpre", style={'color': intel_dark_blue}),
                                    html.Span("ting "),
                                    html.Strong("V", style={'color': intel_dark_blue}),
                                    html.Span("ision-"),
                                    html.Strong("L", style={'color': intel_dark_blue}),
                                    html.Span("anguage "),
                                    html.Strong("T", style={'color': intel_light_blue}),
                                    html.Span("ransformers")
                                ],
                                style={"font-size": "25px"}
                            ),
                        ],
                    )
                ],
                style={ "text-align": "center", "margin": "2%"},
            ),


            # Example selector
            html.Div(
                [html.Label("Example ID", className="plot-label", style={"margin-right": "10px"}),
                 dcc.Input(id="ex_selector", type="number", value=0, min=0, max=n_examples-1),
                 html.Span(" - Model prediction ", className="plot-label", style={"font-weight": "300"}),
                 html.Label("", id="accuracy", className="plot-label", style={"margin-right": "70px"}),
                 html.Button('Add example', id='add_ex_toggle', style={"padding": "0 15px"}),
                 
                 # Add example
                 html.Div([
                     html.Div(
                        [html.Div([
                            html.Label("Image", className="plot-label", style={"margin-right": "10px"}),
                            dcc.Input(
                                id="new_ex_img",
                                placeholder="Enter URL or path",
                                style={"width": "80%", "max-width": "600px", "margin": "5px"})
                        ]), html.Div([
                            html.Label("Text", className="plot-label", style={"margin-right": "10px"}),
                            dcc.Input(id="new_ex_txt",
                                      placeholder="Enter text",
                                      style={"width": "80%", "max-width": "600px", "margin": "5px"})
                        ]),
                        html.Button("Add", id="add_ex_btn",
                                    style={"padding": "0 15px", "margin": "5px", "background": "white"})],
                        style={"margin": "5px 10%", "padding": "5px", "background": "#f1f1f1"})],
                 id="add_ex_div",
                 style={"display": "none"})
                ],
                style={"text-align": "center", "margin": "2%"},
            ),


            # Head summary matrix
            html.Div(
                [html.Label("Attention Head Summary", className="plot-label"),
                 dcc.Graph(id="head_summary", config={"displayModeBar": False})],
                style={"display": "inline-block", "margin-right": "5%", "vertical-align": "top"}
            ),


            # TSNE
            html.Div(
                [
                    html.Label("t-SNE Embeddings", id="tsne_title", className="plot-label"),
                    html.Div([
                        dcc.Graph(id="tsne_map", config=logoConfig, clear_on_unhover=True),
                        dcc.Tooltip(id="hover_out", direction='bottom')
                    ], style={"display": "inline-block", "width": "85%"}),
                    html.Div([
                        html.Label("Layer", style={}),
                        html.Div([
                            dcc.Slider(
                                id="layer_slider", min=0, max=n_layers, step=1,
                                value=0, included=False, vertical=True,
                                marks={i: {
                                    "label": str(n_layers - i),
                                    "style": {"margin": "0 0 -10px -41px" if (i < n_layers - 9) else "0 0 -10px -35px"}
                                } for i in range(n_layers + 1)}
                            )
                        ], style={"margin": "-10px -25px -25px 24px", })
                    ], style={"display": "inline-block", "vertical-align": "top",
                              "width": "44px", "height": "450px", "margin": "-10px 0 0 10px",
                              "border": "1px solid lightgray", "border-radius": "5px"}),
                ],
                style={"width": "45%", "display": "inline-block", "vertical-align": "top"}
            ),

            html.Div(
                [html.Label("Attention", id="attn_title", className="plot-label"),
                 html.Hr(style={"margin": "0 15%"})],
                style={"margin": "3% 0 5px 0"}),


            # Attention to
            html.Div(
                [html.Div(
                    [html.Label("Attention to:", className="plot-label", style={"margin-right": "10px"}),
                     html.Label("Image", id="attn2img_label", className="plot-label"),
                     html.Div(
                        daq.ToggleSwitch(id="attn2img_toggle", value=False, size=40),
                        style={"display": "inline-block", "margin": "0 10px", "vertical-align": "top"}
                     ),
                     html.Label("Text", id="attn2txt_label", className="plot-label"),
                     html.Button("\u25B6", id="play_btn", n_clicks=0, style={"display": "none"})],
                    style={"height": "34px", "width": "400px", "text-align": "left",
                           "display": "inline-block", "padding-left": "100px"}
                 ),
                 html.Div(id="attn2img", children=[  # image
                    html.Div(
                        dcc.Graph(id="ex_image", config={"displayModeBar": False}, style={"height": "450px"}),
                        style={"width": "40%", "position": "absolute"}),
                    html.Div(
                        [dcc.Graph(id="ex_img_token_overlay", config={"displayModeBar": False})],
                        style={"width": "40%", "position": "absolute", "height": "450px"}),
                 ]),
                 html.Div(id="attn2txt", children=[  # text
                    html.Div(
                        [dcc.Graph(id="ex_text", config={"displayModeBar": False})],
                        style={"width": "40%", "position": "absolute", "height": "450px"})
                 ], style={"display": "none", "z-index": "-1"})
                ],
                style={"width": "40%", "display": "inline-block", "vertical-align": "top"}
            ),


            # Attention from
            html.Div(
                [html.Div([html.Label("Attention from Text", className="plot-label", style={"height": "26px"})]),
                 html.Div(  # text display area
                    [dcc.Graph(id="ex_sm_text", config={"displayModeBar": False}, style={"height": "174px"})]
                 ),
                 html.Div([  # label
                    html.Hr(style={"height": "1px", "margin": "0"}),
                    html.Label("Attention from Image", className="plot-label", style={"height": "26px"})
                 ]),
                 html.Div(  # image display area
                    [dcc.Graph(id="ex_sm_image", config={"displayModeBar": False},
                               style={"width": "37%", "height": "256px", "margin-left": "4%", "position": "absolute"}),
                     dcc.Graph(id="ex_img_attn_overlay", config={"displayModeBar": False},
                               style={"width": "37%", "height": "256px", "margin-left": "4%", "position": "absolute"})]
                )],
                style={"height": "484px", "width": "45%", "display": "inline-block", "margin-left": "15px", "vertical-align": "top"},
            ),
            # Movie control (hidden)
            html.Div(
                [dcc.Interval(id="auto_stepper", interval=2500, n_intervals=0, disabled=True)],
                style={"display": "none"}
            ),

            # Checkbox for smoothing
            html.Div(
                [dcc.Checklist(
                    id="map_smooth_checkbox",
                    options=[{'label': 'Smooth attention map', 'value': 'smooth'}],
                    value=['smooth'],
                    style={"text-align": "center"})],
                style={"width": "45%", "display": "inline-block", "margin": "5px 0 2% 0", "padding-left": "40%"}
            ),
        ],
        style={"textAlign": "center"}
    )

    return layout
