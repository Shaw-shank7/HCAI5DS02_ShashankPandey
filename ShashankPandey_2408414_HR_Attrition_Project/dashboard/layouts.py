from dash import html, dcc
import dash_bootstrap_components as dbc

# (Copy this dict from callbacks.py if you didn’t import it there)
GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "resetViews", "toggleSpikelines", "select", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "hoverCompareCartesian"
    ],
    "toImageButtonOptions": {"format": "png", "filename": "hr-attrition-chart"},
    "responsive": True,
}

def kpi_card(id_, title):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="kpi-title"),
            html.H3(id=id_, className="kpi-value")
        ]),
        className="kpi"
    )

controls = dbc.Card(
    dbc.CardBody([
        html.H6("Filters", className="section-title"),
        dbc.Row([
            dbc.Col([
                html.Label("Department", className="form-label"),
                dcc.Dropdown(
                    id="dept-dd",
                    multi=True,
                    placeholder="All",
                    persistence=True, persistence_type="session",
                    maxHeight=260, searchable=True,
                    className="dd"
                )
            ], md=6),
            dbc.Col([
                html.Label("Job Role", className="form-label"),
                dcc.Dropdown(
                    id="role-dd",
                    multi=True,
                    placeholder="All",
                    persistence=True, persistence_type="session",
                    maxHeight=260, searchable=True,
                    className="dd"
                )
            ], md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col([
                html.Label("OverTime", className="form-label"),
                dcc.Dropdown(
                    id="ot-dd",
                    options=[{"label": x, "value": x} for x in ["Yes", "No"]],
                    multi=True, placeholder="All",
                    persistence=True, persistence_type="session",
                    className="dd"
                )
            ], md=6),
            dbc.Col([
                html.Label("Age Range", className="form-label"),
                dcc.RangeSlider(
                    id="age-slider", min=18, max=60, step=1, value=[18, 60],
                    tooltip={"placement": "bottom"},
                    persistence=True, persistence_type="session",
                ),
            ], md=6),
        ], className="g-3"),
    ]),
    className="controls"
)

layout = dbc.Container([
    dcc.Store(id="store-data"),                              # cached data
    html.Div(id="data-error", className="stat-box"),         # prints loader errors (if any)

    html.H1("HR Attrition Analytics", className="app-title"),

    html.Div(
        [
            kpi_card("kpi-emps", "Total Employees"),
            kpi_card("kpi-attr", "Attrition (Count)"),
            kpi_card("kpi-rate", "Attrition Rate (%)"),
            kpi_card("kpi-income", "Avg Monthly Income"),
        ],
        className="kpi-row"
    ),

    controls,

    dcc.Tabs(id="tabs", value="tab-eda", className="tabs", children=[
        dcc.Tab(label="EDA", value="tab-eda", className="tab", selected_className="tab-selected", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-attr-by-dept", config=GRAPH_CONFIG), type="dot"), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-attr-by-role", config=GRAPH_CONFIG), type="dot"), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-age-dist", config=GRAPH_CONFIG), type="dot"), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-income-dist", config=GRAPH_CONFIG), type="dot"), md=6),
            ], className="g-3"),
        ]),
        dcc.Tab(label="Hypothesis Test", value="tab-hypo", className="tab", selected_className="tab-selected", children=[
            html.Br(),
            html.P("H₀: Attrition is independent of Department (no association).", className="hypothesis"),
            html.P("H₁: Attrition depends on Department (there is an association).", className="hypothesis"),
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-dept-attr-heat", config=GRAPH_CONFIG), type="dot"), md=7),
                dbc.Col([
                    html.Div(id="chi2-summary", className="stat-box"),
                    dcc.Markdown(id="chi2-interpret")
                ], md=5),
            ], className="g-3"),
        ]),
        dcc.Tab(label="Modeling", value="tab-model", className="tab", selected_className="tab-selected", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Classifier", className="form-label"),
                    dcc.Dropdown(
                        id="model-dd",
                        value="Logistic Regression",
                        options=[{"label": x, "value": x} for x in [
                            "Logistic Regression", "Decision Tree", "Random Forest", "SVM"
                        ]],
                        clearable=False,
                        persistence=True, persistence_type="session",
                        className="dd"
                    ),
                ], md=4),
            ], className="g-2"),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-roc", config=GRAPH_CONFIG), type="dot"), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="fig-pr", config=GRAPH_CONFIG), type="dot"), md=6),
            ], className="g-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Loading(html.Div(id="model-metrics"), type="dot"), md=12)
            ]),
        ]),
    ]),
], fluid=True)
