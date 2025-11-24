# src/graphics.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.io as pio
import pandas as pd

from . import model as model_module

pio.templates.default = "plotly_white"

# -------------------------------------------------
# Mapeo de nombres técnicos -> legibles
# -------------------------------------------------
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
    return COLUMN_LABELS.get(colname, colname.replace("_", " ").capitalize())


# -------------------------------------------------
# Pestaña EXPLORACIÓN
# -------------------------------------------------
def layout_exploration(df: pd.DataFrame) -> html.Div:
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Exploración de datos",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Elige una variable numérica y una categórica para explorar el comportamiento de las cancelaciones",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
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
                        ]
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


# -------------------------------------------------
# Pestaña PREDICCIÓN
# -------------------------------------------------
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
                    html.H2(
                        "Predicción de cancelación de reserva",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Introduce los datos de una reserva y pulsa «Predecir»",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
                ],
                className="card",
            ),
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
                                        df["arrival_date_month"].dropna().unique()
                                    )
                                ],
                                value=df["arrival_date_month"].dropna().unique()[0],
                                clearable=False,
                                className="dash-dropdown",
                            ),
                        ],
                        className="predict-column",
                    ),
                ],
                className="card predict-row",
            ),
            html.Div(
                [
                    html.Button("Predecir", id="btn-predict", n_clicks=0),
                    html.Button(
                        "Resetear",
                        id="btn-reset",
                        n_clicks=0,
                        style={"marginLeft": "12px"},
                    ),
                    html.Div(id="prediction-output"),
                ],
                className="card",
            ),
        ]
    )


# -------------------------------------------------
# Pestaña RECOMENDACIONES
# -------------------------------------------------
def layout_recommendations(df: pd.DataFrame) -> html.Div:
    # ---- gráficos estáticos con df ----
    # 1) tasa de cancelación por segmento
    if {"market_segment", "is_canceled"}.issubset(df.columns):
        seg = (
            df.groupby("market_segment")["is_canceled"]
            .mean()
            .reset_index()
            .sort_values("is_canceled", ascending=False)
        )
        seg["cancel_rate_pct"] = seg["is_canceled"] * 100
        fig_seg = px.bar(
            seg,
            x="market_segment",
            y="cancel_rate_pct",
            title="Segmentos con mayor tasa de cancelación",
        )
        fig_seg.update_yaxes(title="Tasa de cancelación")
        fig_seg.update_xaxes(title="Segmento de mercado", tickangle=-25)
    else:
        fig_seg = px.bar()

    # 2) estacionalidad por mes
    if {"arrival_date_month", "is_canceled"}.issubset(df.columns):
        month = (
            df.groupby("arrival_date_month")["is_canceled"]
            .mean()
            .reset_index()
            .sort_values("arrival_date_month")
        )
        month["cancel_rate_pct"] = month["is_canceled"] * 100
        fig_month = px.line(
            month,
            x="arrival_date_month",
            y="cancel_rate_pct",
            markers=True,
            title="Estacionalidad de la cancelación por mes de llegada",
        )
        fig_month.update_yaxes(title="Tasa de cancelación")
        fig_month.update_xaxes(title="Mes de llegada", tickangle=-25)
    else:
        fig_month = px.line()

    # 3) correlación numéricas con cancelación
    if "is_canceled" in df.columns:
        num_cols = df.select_dtypes(include="number")
        corr = num_cols.corr()["is_canceled"].drop("is_canceled")
        corr_abs = corr.abs().sort_values(ascending=False).head(7)
        corr_df = (
            corr_abs.reset_index()
            .rename(columns={"index": "variable", "is_canceled": "importance"})
        )
        corr_df["pretty_name"] = corr_df["variable"].apply(pretty_label)

        fig_corr = px.bar(
            corr_df,
            x="pretty_name",
            y="importance",
            color="pretty_name",  # colores distintos por barra
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Factores numéricos más asociados a la cancelación",
        )
        fig_corr.update_yaxes(
            title="Importancia (|correlación| con la cancelación)",
            title_standoff=15,
        )
        fig_corr.update_xaxes(tickangle=-35)
        fig_corr.update_layout(
            margin=dict(l=80, r=20, t=60, b=80),
            showlegend=False,
        )
    else:
        fig_corr = px.bar()

    # ---- layout ----
    return html.Div(
        [
            # Bloque 1: dónde se concentran
            html.Div(
                [
                    html.H2(
                        "¿Dónde se concentran las cancelaciones?",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Analizamos los segmentos de mercado y la estacionalidad para entender en qué tipo de reservas se concentran más cancelaciones",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(figure=fig_seg),
                                style={"width": "48%", "display": "inline-block"},
                            ),
                            html.Div(
                                dcc.Graph(figure=fig_month),
                                style={
                                    "width": "48%",
                                    "display": "inline-block",
                                    "float": "right",
                                },
                            ),
                        ]
                    ),
                ],
                className="card",
            ),
            # Bloque 2: factores numéricos
            html.Div(
                [
                    html.H2(
                        "Factores que más influyen en la cancelación",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Tomamos las variables numéricas del dataset y medimos su correlación con la cancelación para identificar los principales drivers de riesgo",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
                    dcc.Graph(figure=fig_corr),
                ],
                className="card",
            ),
            # Bloque 3: flipcards por tipo de cliente
            html.Div(
                [
                    html.H2(
                        "Guía rápida por tipo de cliente",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Pasa el ratón por cada tarjeta para ver recomendaciones específicas para cada tipo de cliente",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
                    html.Div(
                        [
                            # Transient
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Cliente Transient"),
                                                    html.P(
                                                        "Reservas individuales, estancias cortas y flexibilidad alta"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Ofrecer upselling a tarifas semirrestrictivas y recordatorios de check-in para asegurar la estancia"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-blue",
                                    )
                                ],
                                className="flip-card",
                            ),
                            # Group
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Clientes Group"),
                                                    html.P(
                                                        "Reservas de grupos, alto impacto cuando cancelan"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Pedir depósitos parciales y fijar fechas límite claras para cambios en el grupo"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-purple",
                                    )
                                ],
                                className="flip-card",
                            ),
                            # Corporate
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Cliente Corporate"),
                                                    html.P(
                                                        "Empresas con acuerdos recurrentes y menor tasa de cancelación"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Diseñar acuerdos de cancelación flexibles pero con visibilidad de demanda para revenue"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-green",
                                    )
                                ],
                                className="flip-card",
                            ),
                            # Online TA
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Segmento Online TA"),
                                                    html.P(
                                                        "Reservas hechas en agencias online, alto volumen y cancelación"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Controlar inventario en OTAs, limitar tarifas totalmente reembolsables y usar ofertas no reembolsables en picos de demanda"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-orange",
                                    )
                                ],
                                className="flip-card",
                            ),
                        ],
                        className="flip-grid",
                    ),
                ],
                className="card",
            ),
            # Bloque 4: otros patrones de riesgo
            html.Div(
                [
                    html.H2(
                        "Patrones de reserva con mayor riesgo de cancelación",
                        style={"textAlign": "left"},
                    ),
                    html.P(
                        "Además del tipo de cliente, algunos patrones de reserva concentran más probabilidad de cancelación",
                        className="app-subtitle",
                        style={"textAlign": "left"},
                    ),
                    html.Div(
                        [
                            # Lead time alto
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Lead time muy alto"),
                                                    html.P(
                                                        "Reservas hechas con mucha antelación"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Programar recordatorios automáticos y ofrecer upgrades a tarifas menos flexibles cerca de la fecha de llegada"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-teal",
                                    )
                                ],
                                className="flip-card",
                            ),
                            # ADR bajo + muchas noches
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("ADR bajo + muchas noches"),
                                                    html.P(
                                                        "Reservas largas a precio muy bajo"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Revisar las condiciones, aplicar mínimos de estancia y políticas de cancelación más estrictas"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-pink",
                                    )
                                ],
                                className="flip-card",
                            ),
                            # Muchas cancelaciones previas
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Muchas cancelaciones previas"),
                                                    html.P(
                                                        "Clientes con historial de cancelar a menudo"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-front",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3("Recomendación"),
                                                    html.P(
                                                        "Solicitar prepago parcial o garantía adicional en futuras reservas"
                                                    ),
                                                ],
                                                className="flip-card-face flip-card-back",
                                            ),
                                        ],
                                        className="flip-card-inner gradient-yellow",
                                    )
                                ],
                                className="flip-card",
                            ),
                        ],
                        className="flip-grid",
                    ),
                ],
                className="card",
            ),
        ]
    )


# -------------------------------------------------
# Layout general
# -------------------------------------------------
def create_layout(df: pd.DataFrame) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("CancelGuard", className="header-title"),
                    html.P(
                        "Predice cancelaciones de reservas y explora tus datos hoteleros",
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


# -------------------------------------------------
# Callbacks
# -------------------------------------------------
def register_callbacks(app, df: pd.DataFrame, ml_model):
    # Exploración
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
        fig.update_yaxes(title="Proporción cancelada")
        fig.update_xaxes(tickangle=-25)
        return fig

    # Predicción + reset en un solo callback
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
        ],
        [Input("btn-predict", "n_clicks"), Input("btn-reset", "n_clicks")],
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
            return dash.no_update

        triggered = ctx.triggered[0]["prop_id"].split(".")[0]

        # RESET
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
                None,
            )

        # PREDICCIÓN
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
            msg = html.P(
                "Error al generar la predicción",
                style={"color": "red"},
            )
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
            )

        etiqueta = "SE CANCELA" if pred == 1 else "NO SE CANCELA"
        color = "red" if pred == 1 else "green"

        warning = None
        if prob >= 0.6:
            warning = html.Div(
                "⚠️ Esta reserva tiene alta probabilidad de cancelarse. Consulta la pestaña «Recomendaciones» para estrategias de reducción",
                className="alert-banner",
            )

        msg = html.Div(
            [
                html.P(
                    f"Predicción: {etiqueta}",
                    style={"color": color, "fontWeight": "bold"},
                ),
                html.P(f"Probabilidad: {prob*100:.2f}%"),
                warning,
            ]
        )

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
        )
