# src/graphics.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.io as pio
import pandas as pd

from . import model as model_module

# Tema blanco limpio para todos los gráficos
pio.templates.default = "plotly_white"


# === Mapeo de nombres técnicos -> nombres legibles ===
COLUMN_LABELS = {
    "is_canceled": "Reserva cancelada",
    "lead_time": "Antelación de la reserva (días)",
    "arrival_date_year": "Año de llegada",
    "arrival_date_week_number": "Semana del año de llegada",
    "arrival_date_day_of_month": "Día de llegada",
    "stays_in_weekend_nights": "Noches en fin de semana",
    "stays_in_week_nights": "Noches entre semana",
    "total_nights": "Noches totales",
    "adr": "Precio medio diario (ADR)",
    "adults": "Número de adultos",
    "children": "Número de niños",
    "babies": "Número de bebés",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas no canceladas",
    "required_car_parking_spaces": "Plazas de parking requeridas",
    "total_of_special_requests": "Peticiones especiales",
    "market_segment": "Segmento de mercado",
    "customer_type": "Tipo de cliente",
    "deposit_type": "Tipo de depósito",
    "hotel": "Tipo de hotel",
    "arrival_date_month": "Mes de llegada",
}


def pretty_label(colname: str) -> str:
    """Devuelve un nombre legible para una columna."""
    return COLUMN_LABELS.get(colname, colname.replace("_", " ").capitalize())


# ======================================================================
#  Layout pestaña EXPLORACIÓN
# ======================================================================
def layout_exploration(df: pd.DataFrame) -> html.Div:
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Variable numérica (eje X):"),
                            dcc.Dropdown(
                                id="numeric-col",
                                options=[
                                    {"label": pretty_label(c), "value": c}
                                    for c in numeric_cols
                                ],
                                value=numeric_cols[0],
                                clearable=False,
                                className="dash-dropdown",
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Variable categórica (agrupación):"),
                            dcc.Dropdown(
                                id="cat-col",
                                options=[
                                    {"label": pretty_label(c), "value": c}
                                    for c in categorical_cols
                                ],
                                value=categorical_cols[0],
                                clearable=False,
                                className="dash-dropdown",
                            ),
                        ],
                        style={
                            "width": "48%",
                            "display": "inline-block",
                            "float": "right",
                        },
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    dcc.Graph(id="hist-cancellations"),
                    dcc.Graph(id="bar-cancellations"),
                ],
                className="card",
            ),
        ]
    )


# ======================================================================
#  Layout pestaña PREDICCIÓN
# ======================================================================
def layout_predictor(df: pd.DataFrame) -> html.Div:

    mean_lead_time = int(df["lead_time"].mean()) if "lead_time" in df.columns else 0
    mean_total_nights = (
        int((df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).mean())
        if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns)
        else 1
    )
    mean_adr = float(df["adr"].mean()) if "adr" in df.columns else 0.0

    return html.Div(
        [
            html.Div(
                [
                    html.H2("Predicción de cancelación de reserva"),
                    html.P(
                        "Introduce los datos de una reserva y pulsa «Predecir».",
                        className="app-subtitle",
                    ),
                ],
                className="card",
            ),

            # Formulario de 2 columnas
            html.Div(
                [
                    html.Div(
                        [
                            # Columna izquierda
                            html.Div(
                                [
                                    html.Label("Antelación de la reserva (días):"),
                                    dcc.Input(
                                        id="input-lead-time",
                                        type="number",
                                        value=mean_lead_time,
                                        step=1,
                                        className="dash-input",
                                    ),

                                    html.Label("Noches totales:"),
                                    dcc.Input(
                                        id="input-total-nights",
                                        type="number",
                                        value=mean_total_nights,
                                        step=1,
                                        className="dash-input",
                                    ),

                                    html.Label("Precio medio diario (ADR):"),
                                    dcc.Input(
                                        id="input-adr",
                                        type="number",
                                        value=round(mean_adr, 2),
                                        step=1,
                                        className="dash-input",
                                    ),

                                    html.Label("Peticiones especiales:"),
                                    dcc.Input(
                                        id="input-special-requests",
                                        type="number",
                                        value=0,
                                        step=1,
                                        className="dash-input",
                                    ),
                                ],
                                className="predict-column",
                            ),

                            # Columna derecha
                            html.Div(
                                [
                                    html.Label("Tipo de hotel:"),
                                    dcc.Dropdown(
                                        id="input-hotel",
                                        options=[
                                            {"label": h, "value": h}
                                            for h in sorted(df["hotel"].dropna().unique())
                                        ],
                                        value=df["hotel"].dropna().unique()[0],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),

                                    html.Label("Tipo de depósito:"),
                                    dcc.Dropdown(
                                        id="input-deposit-type",
                                        options=[
                                            {"label": d, "value": d}
                                            for d in sorted(
                                                df["deposit_type"].dropna().unique()
                                            )
                                        ],
                                        value=df["deposit_type"].dropna().unique()[0],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),

                                    html.Label("Tipo de cliente:"),
                                    dcc.Dropdown(
                                        id="input-customer-type",
                                        options=[
                                            {"label": c, "value": c}
                                            for c in sorted(
                                                df["customer_type"].dropna().unique()
                                            )
                                        ],
                                        value=df["customer_type"].dropna().unique()[0],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),

                                    html.Label("Segmento de mercado:"),
                                    dcc.Dropdown(
                                        id="input-market-segment",
                                        options=[
                                            {"label": m, "value": m}
                                            for m in sorted(
                                                df["market_segment"].dropna().unique()
                                            )
                                        ],
                                        value=df["market_segment"].dropna().unique()[0],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),

                                    html.Label("Mes de llegada:"),
                                    dcc.Dropdown(
                                        id="input-arrival-month",
                                        options=[
                                            {"label": m, "value": m}
                                            for m in sorted(
                                                df["arrival_date_month"]
                                                .dropna()
                                                .unique()
                                            )
                                        ],
                                        value=df["arrival_date_month"]
                                        .dropna()
                                        .unique()[0],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),
                                ],
                                className="predict-column",
                            ),
                        ],
                        className="predict-row",
                    ),
                ],
                className="card",
            ),

            # Botones + Predicción
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "Predecir", id="btn-predict", n_clicks=0
                            ),
                            html.Button(
                                "Resetear", id="btn-reset", n_clicks=0
                            ),
                        ],
                        className="predict-buttons",
                    ),
                    html.Div(id="prediction-output", style={"marginTop": "16px"}),
                ],
                className="card",
            ),
        ]
    )


# ======================================================================
#  Layout pestaña RECOMENDACIONES
# ======================================================================
def layout_recommendations(df: pd.DataFrame) -> html.Div:

    if {"market_segment", "is_canceled"}.issubset(df.columns):
        seg = (
            df.groupby("market_segment")["is_canceled"]
            .mean()
            .reset_index()
            .sort_values("is_canceled", ascending=False)
        )
        seg["cancel_rate_pct"] = (seg["is_canceled"] * 100).round(1)
        top_seg = seg.head(3)
        bullets_segments = [
            html.Li(
                f"{row['market_segment']}: {row['cancel_rate_pct']}% de cancelaciones"
            )
            for _, row in top_seg.iterrows()
        ]
    else:
        bullets_segments = [html.Li("No hay datos suficientes.")]

    return html.Div(
        [
            html.Div(
                [
                    html.H2("Recomendaciones basadas en datos"),
                    html.P(
                        "Acciones sugeridas para reducir cancelaciones.",
                        className="app-subtitle",
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H3("Segmentos más problemáticos"),
                    html.Ul(bullets_segments),
                ],
                className="card",
            ),
        ]
    )


# ======================================================================
#  Layout general con TABS normales
# ======================================================================
def create_layout(df: pd.DataFrame) -> html.Div:

    return html.Div(
        [
            # Overlay de aviso de cancelación (vacío al principio)
            html.Div(id="cancel-overlay"),

            # Banner
            html.Div(
                [
                    html.H1("CancelGuard", className="header-title"),
                    html.P(
                        "Predice cancelaciones de reservas y explora tus datos hoteleros.",
                        className="header-subtitle",
                    ),
                ],
                className="header-banner",
            ),

            dcc.Tabs(
                id="tabs",
                value="tab-explore",
                className="tab-parent",
                children=[
                    dcc.Tab(
                        label="Exploración",
                        value="tab-explore",
                        children=layout_exploration(df),
                    ),
                    dcc.Tab(
                        label="Predicción",
                        value="tab-predict",
                        children=layout_predictor(df),
                    ),
                    dcc.Tab(
                        label="Recomendaciones",
                        value="tab-reco",
                        children=layout_recommendations(df),
                    ),
                ],
            ),
        ],
        className="app-container",
    )


# ======================================================================
#  CALLBACKS
# ======================================================================
def register_callbacks(app, df: pd.DataFrame, ml_model):

    # ---------- Exploración: histograma ----------
    @app.callback(
        Output("hist-cancellations", "figure"),
        Input("numeric-col", "value"),
    )
    def update_hist(numeric_col):
        if numeric_col is None or numeric_col not in df.columns:
            return px.histogram()

        fig = px.histogram(
            df,
            x=numeric_col,
            color="is_canceled",
            nbins=40,
            barmode="overlay",
            title=f"Distribución de {pretty_label(numeric_col)}",
        )
        return fig

    # ---------- Exploración: barras ----------
    @app.callback(
        Output("bar-cancellations", "figure"),
        Input("cat-col", "value"),
    )
    def update_bar(cat_col):
        if cat_col is None or "is_canceled" not in df.columns:
            return px.bar()

        grouped = (
            df.groupby(cat_col)["is_canceled"]
            .mean()
            .reset_index()
            .sort_values("is_canceled", ascending=False)
        )

        fig = px.bar(
            grouped,
            x=cat_col,
            y="is_canceled",
            title=f"Tasa de cancelación por {pretty_label(cat_col)}",
        )
        return fig

    # ==================================================================
    # PREDICCIÓN + RESET + OVERLAY
    # ==================================================================
    @app.callback(
        [
            Output("input-lead-time", "value"),
            Output("input-total-nights", "value"),
            Output("input-adr", "value"),
            Output("input-special-requests", "value"),
            Output("input-hotel", "value"),
            Output("input-deposit-type", "value"),
            Output("input-customer-type", "value"),
            Output("input-market-segment", "value"),
            Output("input-arrival-month", "value"),
            Output("prediction-output", "children"),
            Output("cancel-overlay", "children"),
        ],
        [
            Input("btn-predict", "n_clicks"),
            Input("btn-reset", "n_clicks"),
        ],
        [
            State("input-lead-time", "value"),
            State("input-total-nights", "value"),
            State("input-adr", "value"),
            State("input-special-requests", "value"),
            State("input-hotel", "value"),
            State("input-deposit-type", "value"),
            State("input-customer-type", "value"),
            State("input-market-segment", "value"),
            State("input-arrival-month", "value"),
        ],
        prevent_initial_call=True,
    )
    def predict_or_reset(
        click_predict,
        click_reset,
        lead_time,
        total_nights,
        adr,
        special_requests,
        hotel,
        deposit_type,
        customer_type,
        market_segment,
        arrival_month,
    ):

        ctx = dash.callback_context
        if not ctx.triggered:
            return (
                lead_time,
                total_nights,
                adr,
                special_requests,
                hotel,
                deposit_type,
                customer_type,
                market_segment,
                arrival_month,
                dash.no_update,
                dash.no_update,
            )

        triggered = ctx.triggered[0]["prop_id"].split(".")[0]

        # ============ RESET ============
        if triggered == "btn-reset":
            mean_lead_time = int(df["lead_time"].mean())
            mean_total_nights = int(
                (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).mean()
            )
            mean_adr = float(df["adr"].mean())

            return (
                mean_lead_time,
                mean_total_nights,
                round(mean_adr, 2),
                0,
                df["hotel"].dropna().unique()[0],
                df["deposit_type"].dropna().unique()[0],
                df["customer_type"].dropna().unique()[0],
                df["market_segment"].dropna().unique()[0],
                df["arrival_date_month"].dropna().unique()[0],
                None,  # sin mensaje
                None,  # sin overlay
            )

        # ============ PREDICCIÓN ============
        data = {
            "lead_time": lead_time,
            "total_nights": total_nights,
            "adr": adr,
            "total_of_special_requests": special_requests,
            "hotel": hotel,
            "deposit_type": deposit_type,
            "customer_type": customer_type,
            "market_segment": market_segment,
            "arrival_date_month": arrival_month,
        }

        try:
            pred, prob = model_module.predict_cancellation(ml_model, data)
        except Exception:
            msg = html.P("Error al generar la predicción.", style={"color": "red"})
            return (
                lead_time,
                total_nights,
                adr,
                special_requests,
                hotel,
                deposit_type,
                customer_type,
                market_segment,
                arrival_month,
                msg,
                None,
            )

        etiqueta = "SE CANCELA" if pred == 1 else "NO SE CANCELA"
        color = "red" if pred == 1 else "green"

        # Mensaje principal dentro de la tarjeta
        msg_children = [
            html.P(
                f"Predicción: {etiqueta}",
                style={"color": color, "fontWeight": "bold", "fontSize": "18px"},
            ),
            html.P(f"Probabilidad: {prob*100:.2f}%"),
        ]

        overlay_children = None

        if pred == 1:
            # Aviso naranja dentro de la tarjeta
            msg_children.append(
                html.Div(
                    "⚠️ Esta reserva tiene alta probabilidad de cancelarse. "
                    "Consulta la pestaña «Recomendaciones» para estrategias de reducción.",
                    style={
                        "marginTop": "12px",
                        "padding": "10px 14px",
                        "borderRadius": "8px",
                        "backgroundColor": "#fff4e6",
                        "border": "1px solid #ffcc99",
                        "color": "#b55300",
                        "fontWeight": "500",
                    },
                )
            )

            # Overlay a pantalla completa (animado por CSS)
            overlay_children = html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "⚠️ Alta probabilidad de cancelación",
                                className="overlay-title",
                            ),
                            html.Div(
                                "Revisa la pestaña «Recomendaciones» para actuar sobre esta reserva.",
                                className="overlay-text",
                            ),
                        ],
                        className="cancel-overlay-box",
                    )
                ],
                className="cancel-overlay",
            )

        msg = html.Div(msg_children)

        return (
            lead_time,
            total_nights,
            adr,
            special_requests,
            hotel,
            deposit_type,
            customer_type,
            market_segment,
            arrival_month,
            msg,
            overlay_children,
        )
