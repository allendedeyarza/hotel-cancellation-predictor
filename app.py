# app.py
import dash
from src import etl, graphics, model


def create_app():
    print("➡️ Cargando datos...")
    df = etl.load_data()
    print("✅ Datos cargados:", df.shape)

    print("➡️ Entrenando / cargando modelo...")
    ml_model = model.load_model()
    print("✅ Modelo listo.")

    print("➡️ Creando app de Dash...")
    app = dash.Dash(__name__)
    app.title = "CancelGuard"

    print("➡️ Creando layout...")
    app.layout = graphics.create_layout(df)
    print("✅ Layout creado.")

    print("➡️ Registrando callbacks...")
    graphics.register_callbacks(app, df, ml_model)
    print("✅ Callbacks registrados.")

    return app


app = create_app()
server = app.server  # para despliegues futuros


if __name__ == "__main__":
    print("🚀 Levantando servidor en http://127.0.0.1:8060 ...")
    app.run_server(debug=False, port=8060)
