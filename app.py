"""
BigMart Sales — Streamlit App con Agente Conversacional
PI Consulting Data Scientist Challenge

Instalar dependencias:
    pip install streamlit pandas numpy scikit-learn xgboost lightgbm plotly anthropic

Correr:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BigMart Sales — PI Challenge",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
    .chat-message { padding: 14px 18px; border-radius: 12px; margin: 8px 0; line-height: 1.6; }
    .user-msg {
        background: #1565C0;
        color: #ffffff;
        border-left: 4px solid #82B1FF;
        font-weight: 500;
    }
    .bot-msg {
        background: #2b2b2b;
        color: #e0e0e0;
        border-left: 4px solid #7C4DFF;
    }
    .bot-msg code {
        background: #3a3a3a;
        color: #80CBC4;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.88em;
    }
    .bot-msg pre {
        background: #222222;
        color: #A5D6A7;
        padding: 12px;
        border-radius: 8px;
        border-left: 3px solid #7C4DFF;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess data ────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    train = pd.read_csv("Train_BigMart.csv")
    test  = pd.read_csv("Test_BigMart.csv")

    train["split"] = "train"
    test["split"]  = "test"
    test["Item_Outlet_Sales"] = np.nan

    df = pd.concat([train, test], ignore_index=True)

    # Limpieza
    fat_map = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)
    non_food = ["Health and Hygiene", "Household", "Others"]
    df.loc[df["Item_Type"].isin(non_food), "Item_Fat_Content"] = "Non-Edible"

    item_weight_mean = df.groupby("Item_Identifier")["Item_Weight"].transform("mean")
    df["Item_Weight"] = df["Item_Weight"].fillna(item_weight_mean).fillna(df["Item_Weight"].mean())

    outlet_size_mode = df.groupby("Outlet_Type")["Outlet_Size"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Small"
    )
    df["Outlet_Size"] = df.apply(
        lambda r: outlet_size_mode[r["Outlet_Type"]] if pd.isna(r["Outlet_Size"]) else r["Outlet_Size"], axis=1
    )

    item_vis_mean = df.groupby("Item_Identifier")["Item_Visibility"].transform(
        lambda x: x[x > 0].mean() if (x > 0).any() else x.mean()
    )
    df.loc[df["Item_Visibility"] == 0, "Item_Visibility"] = item_vis_mean[df["Item_Visibility"] == 0]

    # Features
    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
    item_mean_vis = df.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
    df["Item_Visibility_MeanRatio"] = df["Item_Visibility"] / item_mean_vis.replace(0, np.nan)

    food_map = {
        "Dairy": "Food", "Soft Drinks": "Drinks", "Meat": "Food",
        "Fruits and Vegetables": "Food", "Household": "Non-Consumable",
        "Baking Goods": "Food", "Snack Foods": "Food", "Frozen Foods": "Food",
        "Breakfast": "Food", "Health and Hygiene": "Non-Consumable",
        "Hard Drinks": "Drinks", "Canned": "Food", "Breads": "Food",
        "Starchy Foods": "Food", "Others": "Non-Consumable", "Seafood": "Food"
    }
    df["Item_Category"] = df["Item_Type"].map(food_map)
    df["MRP_Tier"] = pd.qcut(df["Item_MRP"], q=4, labels=["Budget", "Mid", "Premium", "Luxury"])
    df["Item_Prefix"] = df["Item_Identifier"].str[:2]

    train_clean = df[df["split"] == "train"].copy()
    test_clean  = df[df["split"] == "test"].copy()
    return train_clean, test_clean, df

@st.cache_resource
def train_models(train_df):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb

    cat_cols = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier",
                "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
                "Item_Category", "MRP_Tier", "Item_Prefix"]

    le = LabelEncoder()
    train_enc = train_df.copy()
    for col in cat_cols:
        train_enc[col + "_enc"] = le.fit_transform(train_enc[col].astype(str))

    FEATURES = [
        "Item_Weight", "Item_Fat_Content_enc", "Item_Visibility",
        "Item_Visibility_MeanRatio", "Item_MRP", "Item_Type_enc",
        "Outlet_Age", "Outlet_Size_enc", "Outlet_Location_Type_enc",
        "Outlet_Type_enc", "Item_Category_enc", "MRP_Tier_enc", "Item_Prefix_enc"
    ]

    X = train_enc[FEATURES].fillna(0)
    y = train_enc["Item_Outlet_Sales"]

    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X, y)

    feat_importance = pd.Series(lgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return lgb_model, feat_importance, FEATURES, le, cat_cols

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    col_logo1, col_logo2 = st.columns([1, 1])
    with col_logo1:
        st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=70)
    with col_logo2:
        st.image("images/pi-consulting-logo-dark.png", width=90)
    st.title("BigMart Sales")
    st.caption("PI Consulting — DS Challenge")
    st.divider()

    page = st.radio(
        "Navegación",
        ["📊 Dashboard EDA", "🤖 Modelo & Predicciones", "💬 Chat Agente"],
        index=0
    )

    st.divider()

    # API Key para el chat
    if page == "💬 Chat Agente":
        st.subheader("🔑 Anthropic API Key")
        api_key = st.text_input("Tu API Key", type="password",
                                help="Obtené tu key en console.anthropic.com")
        st.caption("La key no se guarda ni se envía a ningún servidor propio.")
    else:
        api_key = None

    st.divider()
    st.caption("v1.0 · Jimena Fioni · 2026")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Cargando datos..."):
    train_df, test_df, full_df = load_and_preprocess()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard EDA":
    st.title("📊 Exploración de Datos — BigMart Sales")
    st.caption(f"Dataset: {len(train_df):,} registros de entrenamiento · {train_df['Item_Identifier'].nunique()} productos · {train_df['Outlet_Identifier'].nunique()} tiendas")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ventas totales", f"${train_df['Item_Outlet_Sales'].sum():,.0f}")
    with col2:
        st.metric("Venta media por ítem", f"${train_df['Item_Outlet_Sales'].mean():,.0f}")
    with col3:
        st.metric("Mejor outlet", train_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().idxmax())
    with col4:
        st.metric("Producto top (MRP)", f"${train_df['Item_MRP'].max():.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Variable objetivo", "🏪 Análisis de tiendas", "🛍️ Análisis de productos"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(train_df, x="Item_Outlet_Sales", nbins=60,
                               title="Distribución de ventas",
                               color_discrete_sequence=["#667eea"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(train_df, x=np.log1p(train_df["Item_Outlet_Sales"]), nbins=60,
                               title="Distribución de log(1 + ventas)",
                               color_discrete_sequence=["#764ba2"])
            fig.update_xaxes(title="log(1 + Sales)")
            st.plotly_chart(fig, use_container_width=True)

        # MRP vs Sales
        fig = px.scatter(train_df.sample(2000, random_state=42),
                         x="Item_MRP", y="Item_Outlet_Sales",
                         color="Outlet_Type", opacity=0.6,
                         title="Item_MRP vs Ventas por tipo de tienda",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            outlet_sales = (train_df.groupby(["Outlet_Identifier", "Outlet_Type"])
                            ["Item_Outlet_Sales"].sum().reset_index()
                            .sort_values("Item_Outlet_Sales", ascending=True))
            fig = px.bar(outlet_sales, x="Item_Outlet_Sales", y="Outlet_Identifier",
                         color="Outlet_Type", orientation="h",
                         title="Ventas totales por tienda",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(train_df, x="Outlet_Type", y="Item_Outlet_Sales",
                         color="Outlet_Type",
                         title="Distribución de ventas por tipo de tienda",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_xaxes(tickangle=20)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(train_df, x="Outlet_Location_Type", y="Item_Outlet_Sales",
                         color="Outlet_Location_Type",
                         title="Ventas por tipo de ubicación",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            outlet_age = train_df.copy()
            outlet_age["Outlet_Age"] = 2013 - outlet_age["Outlet_Establishment_Year"]
            fig = px.scatter(outlet_age.groupby("Outlet_Identifier").agg(
                Outlet_Age=("Outlet_Age", "first"),
                Total_Sales=("Item_Outlet_Sales", "sum"),
                Outlet_Type=("Outlet_Type", "first")
            ).reset_index(),
                x="Outlet_Age", y="Total_Sales", color="Outlet_Type",
                size="Total_Sales", text="Outlet_Identifier",
                title="Antigüedad de tienda vs Ventas totales",
                color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            item_sales = (train_df.groupby("Item_Type")["Item_Outlet_Sales"]
                          .median().sort_values(ascending=True).reset_index())
            fig = px.bar(item_sales, x="Item_Outlet_Sales", y="Item_Type", orientation="h",
                         title="Ventas medianas por tipo de producto",
                         color="Item_Outlet_Sales",
                         color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(train_df, x="Item_Fat_Content", y="Item_Outlet_Sales",
                         color="Item_Fat_Content",
                         title="Ventas por contenido de grasa",
                         color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        heat_data = train_df.groupby(["Item_Type", "Outlet_Type"])["Item_Outlet_Sales"].median().reset_index()
        heat_pivot = heat_data.pivot(index="Item_Type", columns="Outlet_Type", values="Item_Outlet_Sales")
        fig = px.imshow(heat_pivot, color_continuous_scale="Viridis",
                        title="Mapa de calor: Ventas medianas por tipo de producto y tienda",
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODELO & PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modelo & Predicciones":
    st.title("🤖 Modelo de Predicción — XGBoost + RandomizedSearchCV")

    with st.spinner("Entrenando modelo..."):
        try:
            model, feat_importance, features, le, cat_cols = train_models(train_df)
            model_loaded = True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.info("Asegurate de tener instalado: `pip install xgboost lightgbm`")
            model_loaded = False

    if model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mejor modelo", "XGBoost + RandSearch")
        with col2:
            st.metric("RMSE (CV)", "1,100.52")
        with col3:
            st.metric("R²", "0.5871")
        with col4:
            st.metric("Registros train", f"{len(train_df):,}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🔮 Predictor interactivo", "📋 Resultados en test"])

        with tab1:
            fig = px.bar(feat_importance.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
                         x="Importance", y="Feature",
                         orientation="h", color="Importance",
                         color_continuous_scale="Viridis",
                         title="Importancia de variables (LightGBM — referencia)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **Item_MRP** es el predictor más poderoso, seguido por el tipo de tienda y la antigüedad.")
            st.caption("ℹ️ El gráfico usa LightGBM entrenado en tiempo real para velocidad. El mejor modelo del análisis completo es **XGBoost + RandomizedSearchCV** (RMSE 1,100.52 · R² 0.5871) — sus predicciones están en la tab Resultados.")

        with tab2:
            st.subheader("Predictor de ventas")
            st.caption("Modificá las características del ítem y la tienda para ver la predicción estimada")

            col1, col2 = st.columns(2)
            with col1:
                mrp = st.slider("Item MRP (precio máximo)", 10.0, 270.0, 150.0, 1.0)
                weight = st.slider("Item Weight (gramos)", 4.5, 21.5, 12.0, 0.1)
                visibility = st.slider("Item Visibility", 0.0, 0.35, 0.05, 0.005)
                fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular", "Non-Edible"])
                item_type = st.selectbox("Item Type", sorted(train_df["Item_Type"].unique()))

            with col2:
                outlet_type = st.selectbox("Outlet Type", sorted(train_df["Outlet_Type"].unique()))
                outlet_loc = st.selectbox("Outlet Location", sorted(train_df["Outlet_Location_Type"].unique()))
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
                outlet_year = st.selectbox("Outlet Establishment Year", sorted(train_df["Outlet_Establishment_Year"].unique(), reverse=True))

            if st.button("🔮 Predecir ventas", type="primary"):
                from sklearn.preprocessing import LabelEncoder
                food_map = {
                    "Dairy": "Food", "Soft Drinks": "Drinks", "Meat": "Food",
                    "Fruits and Vegetables": "Food", "Household": "Non-Consumable",
                    "Baking Goods": "Food", "Snack Foods": "Food", "Frozen Foods": "Food",
                    "Breakfast": "Food", "Health and Hygiene": "Non-Consumable",
                    "Hard Drinks": "Drinks", "Canned": "Food", "Breads": "Food",
                    "Starchy Foods": "Food", "Others": "Non-Consumable", "Seafood": "Food"
                }
                # Build a sample row matching training
                sample = pd.DataFrame({
                    "Item_Weight": [weight],
                    "Item_Fat_Content_enc": [le.fit_transform(train_df["Item_Fat_Content"].astype(str)).mean()],
                    "Item_Visibility": [visibility],
                    "Item_Visibility_MeanRatio": [1.0],
                    "Item_MRP": [mrp],
                    "Item_Type_enc": [train_df["Item_Type"].astype("category").cat.codes.mean()],
                    "Outlet_Age": [2013 - outlet_year],
                    "Outlet_Size_enc": [{"Small": 0, "Medium": 1, "High": 2}[outlet_size]],
                    "Outlet_Location_Type_enc": [{"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}[outlet_loc]],
                    "Outlet_Type_enc": [{"Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3}[outlet_type]],
                    "Item_Category_enc": [{"Food": 0, "Drinks": 1, "Non-Consumable": 2}.get(food_map.get(item_type, "Food"), 0)],
                    "MRP_Tier_enc": [0],
                    "Item_Prefix_enc": [0],
                })

                pred = model.predict(sample[features])[0]
                pred = max(0, pred)

                st.success(f"### 💰 Ventas estimadas: **${pred:,.2f}**")

                # Contexto
                p25, p50, p75 = train_df["Item_Outlet_Sales"].quantile([0.25, 0.5, 0.75])
                if pred < p25:
                    st.caption(f"📉 Por debajo del percentil 25 (${p25:,.0f})")
                elif pred < p50:
                    st.caption(f"📊 Entre percentil 25-50 (mediana: ${p50:,.0f})")
                elif pred < p75:
                    st.caption(f"📈 Entre percentil 50-75")
                else:
                    st.caption(f"🚀 Top 25% de ventas")

        with tab3:
            st.subheader("Predicciones sobre el test set")
            try:
                preds_df = pd.read_csv("predictions_test.csv")
                st.dataframe(preds_df.head(50), use_container_width=True)

                pred_col = "Item_Outlet_Sales" if "Item_Outlet_Sales" in preds_df.columns else preds_df.columns[-1]
                fig = px.histogram(preds_df, x=pred_col, nbins=50,
                                   title="Distribución de predicciones (stack ensemble)",
                                   color_discrete_sequence=["#667eea"])
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("⬇️ Descargar predicciones", preds_df.to_csv(index=False),
                                   "predictions_test.csv", "text/csv")
            except FileNotFoundError:
                st.warning("Primero ejecutá el notebook para generar `predictions_test.csv`.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CHAT AGENTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat Agente":
    st.title("💬 Chat con los datos")
    st.caption("Preguntale al agente cualquier cosa sobre el dataset de BigMart")

    if not api_key:
        st.warning("⚠️ Ingresá tu Anthropic API Key en el panel izquierdo para usar el agente.")
        st.info("Sin API Key podés usar las respuestas pre-calculadas de pandas.")

    # Datos de contexto para el agente
    context_stats = {
        "shape": train_df.shape,
        "columns": list(train_df.columns),
        "numeric_stats": train_df.describe().to_dict(),
        "outlet_types": train_df["Outlet_Type"].value_counts().to_dict(),
        "item_types": train_df["Item_Type"].value_counts().to_dict(),
        "outlet_sales": train_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().to_dict(),
        "outlet_type_sales": train_df.groupby("Outlet_Type")["Item_Outlet_Sales"].median().to_dict(),
        "item_type_sales": train_df.groupby("Item_Type")["Item_Outlet_Sales"].median().to_dict(),
        "fat_content_dist": train_df["Item_Fat_Content"].value_counts().to_dict(),
        "nulls": train_df.isnull().sum().to_dict(),
        "correlations": train_df[["Item_Weight","Item_Visibility","Item_MRP","Item_Outlet_Sales"]].corr()["Item_Outlet_Sales"].to_dict(),
    }

    SYSTEM_PROMPT = f"""Sos un analista de datos experto en Python y pandas, especializado en el dataset BigMart Sales.

El dataset de entrenamiento tiene {context_stats['shape'][0]:,} filas y {context_stats['shape'][1]} columnas.
Columnas: {context_stats['columns']}

Estadísticas clave:
- Ventas (Item_Outlet_Sales): media={train_df['Item_Outlet_Sales'].mean():.0f}, mediana={train_df['Item_Outlet_Sales'].median():.0f}, max={train_df['Item_Outlet_Sales'].max():.0f}
- Tipos de tienda: {context_stats['outlet_types']}
- Ventas medianas por tipo de tienda: {context_stats['outlet_type_sales']}
- Tipos de producto: {list(context_stats['item_types'].keys())}
- Ventas medianas por tipo de producto: {context_stats['item_type_sales']}
- Correlaciones con ventas: {context_stats['correlations']}
- Valores nulos: {context_stats['nulls']}

Cuando el usuario haga preguntas sobre los datos, respondé con:
1. La respuesta directa con números concretos
2. Un código Python/pandas que el usuario puede ejecutar para verificarlo
3. Una interpretación o insight del resultado

Respondé siempre en español. Sé conciso y directo. Si te preguntan algo fuera del dataset, decí que solo podés ayudar con los datos de BigMart."""

    # Preguntas sugeridas
    st.subheader("💡 Preguntas sugeridas")
    col1, col2, col3 = st.columns(3)

    suggested = [
        "¿Qué outlet tiene las mayores ventas totales?",
        "¿Cuáles son los 5 tipos de producto con más ventas?",
        "¿Hay correlación entre el MRP y las ventas?",
        "¿Cómo afecta el tamaño de la tienda a las ventas?",
        "¿Qué tipo de tienda vende más en promedio?",
        "¿Cuántos productos tienen visibilidad 0?",
    ]

    for i, q in enumerate(suggested):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_input = q

    st.divider()

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Historial de mensajes con st.chat_message (renderiza markdown y código)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Input
    default_val = st.session_state.pop("chat_input", "")
    user_input = st.chat_input("Preguntá sobre los datos...", key="chat_input_widget")

    if not user_input and default_val:
        user_input = default_val

    if user_input:
        # Mostrar mensaje del usuario
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Analizando..."):
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    )
                    answer = response.content[0].text
                except Exception as e:
                    answer = f"❌ Error con la API: {e}\n\nVerificá que tu API Key sea válida."
            else:
                # Respuestas pre-calculadas sin API
                q_lower = user_input.lower()
                if "outlet" in q_lower and ("mayor" in q_lower or "más" in q_lower):
                    best = train_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().idxmax()
                    best_val = train_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().max()
                    answer = (
                        f"**{best}** tiene las mayores ventas totales con **${best_val:,.0f}**.\n\n"
                        f"```python\n"
                        f"train.groupby('Outlet_Identifier')['Item_Outlet_Sales']\\\n"
                        f"     .sum().sort_values(ascending=False).head()\n"
                        f"```"
                    )
                elif "mrp" in q_lower and "correlaci" in q_lower:
                    corr = train_df["Item_MRP"].corr(train_df["Item_Outlet_Sales"])
                    answer = (
                        f"La correlación entre **Item_MRP** y las ventas es **{corr:.3f}** "
                        f"— la más alta entre las variables numéricas.\n\n"
                        f"```python\n"
                        f"train[['Item_MRP', 'Item_Outlet_Sales']].corr()\n"
                        f"```"
                    )
                elif "visibilidad" in q_lower and "0" in q_lower:
                    n = (train_df["Item_Visibility"] == 0).sum()
                    answer = (
                        f"Hay **{n} registros** con visibilidad igual a 0 "
                        f"({n/len(train_df)*100:.1f}% del dataset). "
                        f"Son errores de registro — se imputan con la media por ítem.\n\n"
                        f"```python\n"
                        f"(train['Item_Visibility'] == 0).sum()\n"
                        f"```"
                    )
                elif "tipo" in q_lower and "tienda" in q_lower:
                    stats = train_df.groupby("Outlet_Type")["Item_Outlet_Sales"].agg(["mean", "median"]).round(0)
                    answer = (
                        f"Ventas por tipo de tienda:\n\n"
                        f"{stats.to_markdown()}\n\n"
                        f"💡 **Supermarket Type3** tiene las mayores ventas promedio.\n\n"
                        f"```python\n"
                        f"train.groupby('Outlet_Type')['Item_Outlet_Sales'].agg(['mean','median'])\n"
                        f"```"
                    )
                elif "producto" in q_lower or "item" in q_lower:
                    top5 = train_df.groupby("Item_Type")["Item_Outlet_Sales"].median().sort_values(ascending=False).head(5)
                    answer = (
                        f"Top 5 tipos de producto por ventas medianas:\n\n"
                        f"{top5.to_markdown()}\n\n"
                        f"```python\n"
                        f"train.groupby('Item_Type')['Item_Outlet_Sales'].median().sort_values(ascending=False).head(5)\n"
                        f"```"
                    )
                else:
                    answer = (
                        f"Para responder preguntas complejas necesito la API Key de Anthropic. "
                        f"Con ella puedo analizar: `{user_input}`\n\n"
                        f"Sin API Key podés usar las preguntas sugeridas de arriba, "
                        f"o explorar el **Dashboard EDA** y el **Predictor interactivo**."
                    )

        # Mostrar respuesta del agente
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button("🗑️ Limpiar chat"):
            st.session_state.messages = []
            st.rerun()
