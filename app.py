# app.py
import dash
from src import etl, graphics, model

# 1. Cargar datos
df = etl.load_data()

# 2. Cargar modelo (si existe)
ml_model = model.load_model()

# 3. Crear app Dash
app = dash.Dash(__name__)
server = app.server  # necesario para Render/gunicorn

# 4. Layout
app.layout = graphics.create_layout(df)

# 5. Callbacks (le pasamos también el modelo)
graphics.register_callbacks(app, df, ml_model)

if __name__ == "__main__":
    app.run_server(debug=True)
