# dash v2+ safe
import dash
from dash import Input, Output  # for clientside wiring
import dash_bootstrap_components as dbc
from layouts import layout  # sidebar + three panes layout (pane-eda / pane-hypo / pane-model)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],  # your assets/style.css auto-loads too
    suppress_callback_exceptions=True,
    title="HR Attrition Dashboard",
)

server = app.server
app.layout = layout

# IMPORTANT: keep this import so @callback functions in callbacks.py get registered
import callbacks  # noqa: F401

# --- Sidebar switcher: show the chosen pane on the right (no server compute) ---
# Uses the existing 'tabs' value ("tab-eda" | "tab-hypo" | "tab-model")
# We update a harmless dummy prop just to trigger on change.
app.clientside_callback(
    """
    function(tabValue){
        const show = (id, on) => {
            const el = document.getElementById(id);
            if (!el) return null;
            el.style.display = on ? "block" : "none";
            return null;
        }
        show("pane-eda",   tabValue === "tab-eda");
        show("pane-hypo",  tabValue === "tab-hypo");
        show("pane-model", tabValue === "tab-model");
        return "";
    }
    """,
    Output("data-error", "title"),   # dummy output prop
    Input("tabs", "value"),
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
