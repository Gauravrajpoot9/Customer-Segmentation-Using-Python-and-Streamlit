"""
Customer Segmentation Dashboard (Streamlit)

Run with:
    pip install streamlit pandas numpy matplotlib plotly scikit-learn wordcloud
    streamlit run streamlit_app.py
"""

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st  # type: ignore[import-not-found]
from sklearn import ensemble, linear_model, metrics, neighbors, svm, tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🛍️", layout="wide")

# ---------------------------------------------------------------------------
# THEME / STYLE
# ---------------------------------------------------------------------------
COLORS = ["#6C5CE7", "#00B894", "#FD79A8", "#FDCB6E", "#0984E3", "#E17055",
          "#00CEC9", "#A29BFE", "#FF7675", "#55EFC4", "#74B9FF", "#FAB1A0"]
px.defaults.color_discrete_sequence = COLORS
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Plasma

# ---- EDIT THESE TWO LINES TO CHANGE THE LOOK -------------------------------------------
# Light background -> use dark text.  Dark background -> use light text.
MAIN_BG = "linear-gradient(160deg, #f6f3ff 0%, #eefaf6 50%, #fff4f8 100%)"
TEXT_COLOR = "#1f1f2e"      # e.g. "#f2f0ff" if you choose a dark MAIN_BG
HEADING_COLOR = "#3b2a8f"   # e.g. "#c9bfff" if you choose a dark MAIN_BG
# -----------------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background: __MAIN_BG__; }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
    [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"],
    .stTabs [data-baseweb="tab"] { color: __TEXT_COLOR__; }
    .hero, .hero h1, .hero p, .kpi, .kpi * { color: #ffffff !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #2d1e6b 0%, #4a2fa0 60%, #6C5CE7 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * { color: #222 !important; }
    .hero { padding: 26px 32px; border-radius: 20px; color: white; margin-bottom: 18px;
            background: linear-gradient(120deg, #6C5CE7, #FD79A8 55%, #FDCB6E);
            box-shadow: 0 8px 24px rgba(108,92,231,.35); }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p { margin: 4px 0 0 0; opacity: .92; }
    .kpi { padding: 18px 20px; border-radius: 18px; color: white; margin-bottom: 12px;
           box-shadow: 0 6px 18px rgba(0,0,0,.15); }
    .kpi .icon { font-size: 1.6rem; }
    .kpi .v { font-size: 1.7rem; font-weight: 800; line-height: 1.2; }
    .kpi .l { font-size: .85rem; opacity: .92; letter-spacing: .5px; text-transform: uppercase; }
    h1, h2, h3 { color: __HEADING_COLOR__; }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        background: white; border-radius: 16px; padding: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.07); }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """.replace("__MAIN_BG__", MAIN_BG).replace("__TEXT_COLOR__", TEXT_COLOR)
       .replace("__HEADING_COLOR__", HEADING_COLOR),
    unsafe_allow_html=True,
)


def hero(title, subtitle=""):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, icon, label, value, c1, c2):
    col.markdown(
        f'<div class="kpi" style="background:linear-gradient(135deg,{c1},{c2})">'
        f'<div class="icon">{icon}</div><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def show(fig, height=400):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


N_PRODUCT_CLUSTERS = 5
N_CUSTOMER_CLUSTERS = 11
CUTOFF = pd.Timestamp("2011-10-01")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STOP_WORDS = {
    "THE", "AND", "FOR", "WITH", "SET", "PINK", "BLUE", "GREEN", "ORANGE", "RED", "WHITE",
    "BLACK", "YELLOW", "PURPLE", "TAG", "LARGE", "SMALL", "MEDIUM", "ASSORTED", "DESIGN", "VINTAGE",
}
LIST_COLS = ["count", "min", "max", "mean", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4"]
FEATURE_COLS = ["mean", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4"]


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(file_or_path):
    df = pd.read_csv(
        file_or_path, encoding="ISO-8859-1",
        dtype={"CustomerID": str, "InvoiceNo": str, "StockCode": str},
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@st.cache_data(show_spinner="Cleaning data (cancellations, duplicates)...")
def clean_data(df_raw):
    df = df_raw.dropna(subset=["CustomerID"]).drop_duplicates().copy()
    df_cleaned = df.copy()
    df_cleaned["QuantityCanceled"] = 0
    entry_to_remove, doubtful_entry = [], []

    pos = df[df["Quantity"] > 0]
    group_idx = pos.groupby(["CustomerID", "StockCode"]).groups
    cancel_rows = df[(df["Quantity"] < 0) & (df["Description"] != "Discount")]

    for index, row in cancel_rows.iterrows():
        key = (row["CustomerID"], row["StockCode"])
        if key not in group_idx:
            doubtful_entry.append(index)
            continue
        cand = pos.loc[group_idx[key]]
        cand = cand[cand["InvoiceDate"] < row["InvoiceDate"]]
        if cand.empty:
            doubtful_entry.append(index)
        elif len(cand) == 1:
            df_cleaned.loc[cand.index[0], "QuantityCanceled"] = -row["Quantity"]
            entry_to_remove.append(index)
        else:
            cand = cand.sort_index(ascending=False)
            for ind, val in cand.iterrows():
                if val["Quantity"] >= -row["Quantity"]:
                    df_cleaned.loc[ind, "QuantityCanceled"] = -row["Quantity"]
                    entry_to_remove.append(index)
                    break

    df_cleaned = df_cleaned.drop(entry_to_remove + doubtful_entry, errors="ignore")
    df_cleaned["TotalPrice"] = df_cleaned["UnitPrice"] * (
        df_cleaned["Quantity"] - df_cleaned["QuantityCanceled"]
    )
    return df_cleaned, {"entry_to_remove": len(entry_to_remove), "doubtful_entry": len(doubtful_entry)}


@st.cache_data(show_spinner="Preparing sales table...")
def prepare_sales(df_cleaned):
    s = df_cleaned[(df_cleaned["TotalPrice"] > 0) & (df_cleaned["Quantity"] > 0)].copy()
    s["Month"] = s["InvoiceDate"].dt.to_period("M").astype(str)
    s["Weekday"] = s["InvoiceDate"].dt.day_name()
    s["Hour"] = s["InvoiceDate"].dt.hour
    return s


# ---------------------------------------------------------------------------
# PRODUCTS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Extracting keywords...")
def extract_keywords(descriptions, min_count=13):
    counter = Counter()
    for desc in descriptions:
        if not isinstance(desc, str):
            continue
        for w in set(re.findall(r"[A-Z]{3,}", desc.upper())):
            if w not in STOP_WORDS:
                counter[w] += 1
    return sorted([(w, c) for w, c in counter.items() if c >= min_count], key=lambda x: x[1], reverse=True)


@st.cache_data(show_spinner="Clustering products...")
def cluster_products(df_cleaned, keywords, n_clusters=N_PRODUCT_CLUSTERS):
    # plain tuple of str: pandas 3 returns an ArrowStringArray that Streamlit cannot hash
    products = tuple(str(x) for x in df_cleaned["Description"].dropna().unique())
    X = pd.DataFrame(index=range(len(products)))
    for key, _ in keywords:
        X[key] = [int(key in p.upper()) for p in products]
    threshold = [0, 1, 2, 3, 5, 10]
    mean_price = df_cleaned.groupby("Description")["UnitPrice"].mean()
    prices = mean_price.reindex(list(products)).fillna(0).values
    for i, t in enumerate(threshold):
        col = f".>{t}" if i == len(threshold) - 1 else f"{t}<.<{threshold[i + 1]}"
        hi = np.inf if i == len(threshold) - 1 else threshold[i + 1]
        X[col] = ((prices > t) & (prices <= hi)).astype(int)
    matrix = X.to_numpy()
    best_score, best_clusters = -1, None
    for seed in range(5):
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=seed)
        clusters = km.fit_predict(matrix)
        score = silhouette_score(matrix, clusters)
        if score > best_score:
            best_score, best_clusters = score, clusters
    return products, matrix, best_clusters, best_score


# ---------------------------------------------------------------------------
# CUSTOMERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building baskets...")
def build_baskets(df_cleaned, products, clusters):
    df = df_cleaned.copy()
    df["categ_product"] = df["Description"].map(dict(zip(products, clusters)))
    for i in range(N_PRODUCT_CLUSTERS):
        df[f"categ_{i}"] = df["TotalPrice"].where(df["categ_product"] == i, 0).clip(lower=0)
    # Keep the real datetime (no int64 round-trip: newer pandas uses datetime64[us],
    # which turned every date into 1970 and left the test set empty).
    agg = {"TotalPrice": "sum", "InvoiceDate": "first"}
    agg.update({f"categ_{i}": "sum" for i in range(N_PRODUCT_CLUSTERS)})
    basket = df.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(agg)
    basket = basket.rename(columns={"TotalPrice": "Basket Price"})
    return basket[basket["Basket Price"] > 0]


def per_user(basket, first_last=True):
    t = basket.groupby("CustomerID")["Basket Price"].agg(["count", "min", "max", "mean", "sum"])
    for i in range(N_PRODUCT_CLUSTERS):
        col = f"categ_{i}"
        t[col] = basket.groupby("CustomerID")[col].sum() / t["sum"] * 100
    t = t.reset_index()
    if first_last:
        last_date = basket["InvoiceDate"].max().date()
        first = basket.groupby("CustomerID")["InvoiceDate"].min().dt.date
        last = basket.groupby("CustomerID")["InvoiceDate"].max().dt.date
        t["FirstPurchase"] = t["CustomerID"].map(lambda c: (last_date - first[c]).days)
        t["LastPurchase"] = t["CustomerID"].map(lambda c: (last_date - last[c]).days)
    return t


@st.cache_resource(show_spinner="Clustering customers...")
def cluster_customers(basket_train):
    users = per_user(basket_train)
    matrix = users[LIST_COLS].to_numpy()
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    km = KMeans(init="k-means++", n_clusters=N_CUSTOMER_CLUSTERS, n_init=20, random_state=0)
    labels = km.fit_predict(scaled)
    users["cluster"] = labels
    return users, scaler, scaled, km, silhouette_score(scaled, labels)


@st.cache_resource(show_spinner="Training classifiers (grid search)...")
def train_classifiers(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
    specs = {
        "Support Vector Machine": (svm.LinearSVC(max_iter=5000), {"C": np.logspace(-2, 2, 5)}),
        "Logistic Regression": (linear_model.LogisticRegression(max_iter=2000), {"C": np.logspace(-2, 2, 5)}),
        "k-Nearest Neighbors": (neighbors.KNeighborsClassifier(), {"n_neighbors": np.arange(1, 30, 2)}),
        "Decision Tree": (tree.DecisionTreeClassifier(random_state=0),
                          {"criterion": ["entropy", "gini"], "max_features": ["sqrt", "log2"]}),
        "Random Forest": (ensemble.RandomForestClassifier(random_state=0),
                          {"n_estimators": [20, 60, 100], "criterion": ["entropy", "gini"]}),
        "Gradient Boosting": (ensemble.GradientBoostingClassifier(random_state=0),
                              {"n_estimators": [20, 50, 100]}),
    }
    models, results = {}, []
    for name, (clf, grid) in specs.items():
        gs = GridSearchCV(clf, grid, cv=5).fit(X_train, Y_train)
        pred = gs.predict(X_test)
        models[name] = gs
        results.append({"Classifier": name, "Best params": str(gs.best_params_),
                        "Accuracy (%)": round(100 * metrics.accuracy_score(Y_test, pred), 2)})
    voting = ensemble.VotingClassifier(
        estimators=[
            ("rf", ensemble.RandomForestClassifier(random_state=0, **models["Random Forest"].best_params_)),
            ("gb", ensemble.GradientBoostingClassifier(random_state=0, **models["Gradient Boosting"].best_params_)),
            ("knn", neighbors.KNeighborsClassifier(**models["k-Nearest Neighbors"].best_params_)),
        ],
        voting="soft",
    ).fit(X_train, Y_train)
    results.append({"Classifier": "Voting (RF + GB + kNN)", "Best params": "-",
                    "Accuracy (%)": round(100 * metrics.accuracy_score(Y_test, voting.predict(X_test)), 2)})
    models["Voting"] = voting
    return models, pd.DataFrame(results), (X_test, Y_test)


def rfm_table(sales):
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    g = sales.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": (snapshot - g["InvoiceDate"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["TotalPrice"].sum(),
    })
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def seg(r):
        if r.R >= 4 and r.F >= 4:
            return "Champions"
        if r.R >= 3 and r.F >= 3:
            return "Loyal"
        if r.R >= 4 and r.F <= 2:
            return "New / Promising"
        if r.R <= 2 and r.F >= 4:
            return "Can't Lose"
        if r.R <= 2 and r.F >= 3:
            return "At Risk"
        if r.R <= 2:
            return "Lost / Hibernating"
        return "Needs Attention"

    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm.reset_index()


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛍️ Segmentation Hub")
    uploaded = st.file_uploader("Upload data.csv", type="csv")
    path = st.text_input("...or path to CSV", value="data.csv")

source = uploaded if uploaded is not None else path
try:
    df_raw = load_data(source)
except Exception as e:
    hero("🛍️ Customer Segmentation Dashboard", "Upload your dataset in the sidebar to begin")
    st.info("Upload the dataset (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, "
            "UnitPrice, CustomerID, Country) or give a valid path in the sidebar.")
    st.caption(f"Details: {e}")
    st.stop()
    raise SystemExit

df_cleaned, clean_info = clean_data(df_raw)
sales = prepare_sales(df_cleaned)

with st.sidebar:
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🌍 Geography", "📦 Products", "💎 RFM Segments", "🔁 Cohort Retention",
         "🧩 ML Segmentation", "🤖 Classification", "🧪 Test (last 2 months)", "🔮 Predict a Customer",
         "🔎 Customer Lookup", "📋 Data Explorer"],
    )
    st.markdown("---")
    st.markdown("### Filters")
    all_countries = sorted(sales["Country"].unique())
    sel_countries = st.multiselect("Countries", all_countries, default=[])
    dmin, dmax = sales["InvoiceDate"].min().date(), sales["InvoiceDate"].max().date()
    dr = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax, key="date_range")
    st.caption("Filters apply to Dashboard, Geography, Products, RFM, Cohort, Lookup & Explorer. "
               "ML pages always use the full dataset.")

# st.date_input returns a single date (not a tuple) or a 1-tuple while the user is still
# picking the range, and an empty tuple if cleared. Handle all of these instead of silently
# falling back to the full range.
if isinstance(dr, (list, tuple)):
    if len(dr) == 2:
        d0, d1 = dr
    elif len(dr) == 1:
        d0, d1 = dr[0], dmax
        st.sidebar.info("Pick an end date to finish the range (showing from start date onwards).")
    else:
        d0, d1 = dmin, dmax
else:
    d0, d1 = dr, dmax
d0, d1 = max(min(d0, d1), dmin), min(max(d0, d1), dmax)

# Compare on real timestamps (inclusive of the whole last day) - faster than .dt.date
mask = (sales["InvoiceDate"] >= pd.Timestamp(d0)) & (sales["InvoiceDate"] < pd.Timestamp(d1) + pd.Timedelta(days=1))
if sel_countries:
    mask &= sales["Country"].isin(sel_countries)
df_f = sales[mask]


def need_data():
    if df_f.empty:
        st.warning("No data for the selected filters.")
        st.stop()
        raise SystemExit


def get_pipeline():
    keywords = extract_keywords(tuple(df_cleaned["Description"].dropna().unique()))
    products, pmatrix, pclusters, pscore = cluster_products(df_cleaned, keywords)
    basket = build_baskets(df_cleaned, products, pclusters)
    train = basket[basket["InvoiceDate"] < CUTOFF].copy()
    test = basket[basket["InvoiceDate"] >= CUTOFF].copy()
    users, scaler, scaled, km, cscore = cluster_customers(train)
    return keywords, products, pmatrix, pclusters, pscore, basket, train, test, users, scaler, scaled, km, cscore


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    hero("🏠 Sales Dashboard", f"{d0} → {d1} • {len(sel_countries) or 'all'} country filter")
    need_data()
    revenue = df_f["TotalPrice"].sum()
    orders = df_f["InvoiceNo"].nunique()
    c = st.columns(6)
    kpi(c[0], "💰", "Revenue", f"£{revenue:,.0f}", "#6C5CE7", "#A29BFE")
    kpi(c[1], "🧾", "Orders", f"{orders:,}", "#00B894", "#55EFC4")
    kpi(c[2], "👥", "Customers", f"{df_f['CustomerID'].nunique():,}", "#FD79A8", "#FF7675")
    kpi(c[3], "🛒", "Avg order value", f"£{revenue / orders:,.0f}", "#E17055", "#FDCB6E")
    kpi(c[4], "📦", "Products", f"{df_f['StockCode'].nunique():,}", "#0984E3", "#74B9FF")
    kpi(c[5], "🌍", "Countries", f"{df_f['Country'].nunique()}", "#00CEC9", "#81ECEC")

    monthly = df_f.groupby("Month").agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
    a, b = st.columns([3, 2])
    with a:
        fig = px.area(monthly, x="Month", y="Revenue", title="📈 Monthly revenue", markers=True)
        fig.update_traces(line_color="#6C5CE7", fillcolor="rgba(108,92,231,.25)")
        show(fig, 380)
    with b:
        fig = px.bar(monthly, x="Month", y="Orders", title="🧾 Orders per month",
                     color="Orders", color_continuous_scale="Tealgrn")
        show(fig, 380)

    a, b = st.columns(2)
    with a:
        top_c = df_f.groupby("Country")["TotalPrice"].sum().nlargest(10).reset_index()
        fig = px.bar(top_c, x="TotalPrice", y="Country", orientation="h", color="TotalPrice",
                     title="🌍 Top 10 countries by revenue", color_continuous_scale="Sunsetdark")
        fig.update_yaxes(autorange="reversed")
        show(fig)
    with b:
        top_p = df_f.groupby("Description")["TotalPrice"].sum().nlargest(10).reset_index()
        fig = px.bar(top_p, x="TotalPrice", y="Description", orientation="h", color="TotalPrice",
                     title="🏆 Top 10 products by revenue", color_continuous_scale="Viridis")
        fig.update_yaxes(autorange="reversed")
        show(fig)

    a, b = st.columns([3, 2])
    with a:
        heat = df_f.pivot_table(index="Weekday", columns="Hour", values="TotalPrice", aggfunc="sum").reindex(WEEKDAYS)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="Turbo",
                        title="🕒 Revenue by weekday & hour", labels=dict(color="£"))
        show(fig, 380)
    with b:
        wd = df_f.groupby("Weekday")["TotalPrice"].sum().reindex(WEEKDAYS).reset_index()
        fig = px.pie(wd, names="Weekday", values="TotalPrice", hole=0.5, title="📅 Revenue share by weekday")
        show(fig, 380)

    basket_p = df_f.groupby("InvoiceNo")["TotalPrice"].sum()
    fig = px.histogram(basket_p[basket_p < basket_p.quantile(0.99)], nbins=60,
                       title="🧺 Basket value distribution (99th percentile cut)",
                       color_discrete_sequence=["#FD79A8"])
    fig.update_layout(showlegend=False, xaxis_title="Basket value (£)")
    show(fig, 340)

elif page == "🌍 Geography":
    hero("🌍 Geography", "Where do the orders and revenue come from?")
    need_data()
    geo = df_f.groupby("Country").agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique"),
                                      Customers=("CustomerID", "nunique")).reset_index()
    metric = st.radio("Metric", ["Revenue", "Orders", "Customers"], horizontal=True)
    fig = px.choropleth(geo, locations="Country", locationmode="country names", color=np.log10(geo[metric] + 1),
                        hover_name="Country", hover_data={metric: True},
                        color_continuous_scale="Rainbow", title=f"{metric} per country (log10 colour scale)")
    fig.update_layout(coloraxis_colorbar_title="log10")
    show(fig, 520)
    a, b = st.columns(2)
    with a:
        fig = px.treemap(geo.nlargest(20, "Revenue"), path=["Country"], values="Revenue",
                         color="Revenue", color_continuous_scale="Spectral", title="Revenue treemap (top 20)")
        show(fig, 420)
    with b:
        st.dataframe(geo.sort_values("Revenue", ascending=False).round(0), use_container_width=True, height=420)

elif page == "📦 Products":
    hero("📦 Products", "Best sellers, keywords and product clusters")
    need_data()
    t1, t2, t3 = st.tabs(["🏆 Best sellers", "🔤 Keywords & clusters", "☁️ Word clouds"])
    with t1:
        by = st.radio("Rank by", ["Revenue", "Quantity"], horizontal=True)
        n = st.slider("How many products", 5, 30, 15)
        g = df_f.groupby("Description").agg(Revenue=("TotalPrice", "sum"), Quantity=("Quantity", "sum")).reset_index()
        g = g.nlargest(n, by)
        fig = px.bar(g, x=by, y="Description", orientation="h", color=by, color_continuous_scale="Turbo",
                     title=f"Top {n} products by {by.lower()}")
        fig.update_yaxes(autorange="reversed")
        show(fig, 120 + 28 * n)
        price = df_f.groupby("Description").agg(P=("UnitPrice", "mean"), Q=("Quantity", "sum"),
                                                R=("TotalPrice", "sum")).reset_index()
        fig = px.scatter(price[price["P"] < 100], x="P", y="Q", size="R", color="R", log_y=True, hover_name="Description",
                         color_continuous_scale="Plasma", title="Unit price vs quantity sold (bubble = revenue)",
                         labels={"P": "Average unit price (£)", "Q": "Quantity"})
        show(fig, 450)

    keywords, products, pmatrix, pclusters, pscore, *_ = get_pipeline()
    with t2:
        kw_df = pd.DataFrame(keywords[:30], columns=["keyword", "occurrences"])
        a, b = st.columns(2)
        with a:
            fig = px.bar(kw_df, x="occurrences", y="keyword", orientation="h", color="occurrences",
                         color_continuous_scale="Sunset", title="Most frequent keywords")
            fig.update_yaxes(autorange="reversed")
            show(fig, 700)
        with b:
            st.metric("Silhouette score", f"{pscore:.3f}")
            counts = pd.Series(pclusters).value_counts().sort_index().reset_index()
            counts.columns = ["cluster", "products"]
            counts["cluster"] = counts["cluster"].astype(str)
            fig = px.pie(counts, names="cluster", values="products", hole=0.45, title="Products per cluster")
            show(fig, 320)
            comp = PCA(n_components=2).fit_transform(pmatrix)
            pdf = pd.DataFrame(comp, columns=["PC1", "PC2"])
            pdf["cluster"] = pclusters.astype(str)
            show(px.scatter(pdf, x="PC1", y="PC2", color="cluster", opacity=0.55, title="PCA of product clusters"), 340)
    with t3:
        prod_series = pd.Series(products).str.upper()
        words = [w for w, _ in keywords]
        cols = st.columns(2)
        for i in range(N_PRODUCT_CLUSTERS):
            sub = prod_series[pclusters == i]
            freq = {w: int(sub.str.contains(w, regex=False).sum()) for w in words}
            freq = {w: v for w, v in freq.items() if v > 0}
            if not freq:
                continue
            wc = WordCloud(width=700, height=350, background_color="white", colormap="plasma").generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Cluster {i}", color="#3b2a8f", fontweight="bold")
            cols[i % 2].pyplot(fig)
            plt.close(fig)

elif page == "💎 RFM Segments":
    hero("💎 RFM Segmentation", "Recency • Frequency • Monetary value of every customer")
    need_data()
    rfm = rfm_table(df_f)
    seg_stats = rfm.groupby("Segment").agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"),
                                           Recency=("Recency", "mean"), Frequency=("Frequency", "mean")).reset_index()
    c = st.columns(4)
    kpi(c[0], "🏅", "Champions", f"{(rfm['Segment'] == 'Champions').sum():,}", "#6C5CE7", "#A29BFE")
    kpi(c[1], "💚", "Loyal", f"{(rfm['Segment'] == 'Loyal').sum():,}", "#00B894", "#55EFC4")
    n_risk = int(rfm["Segment"].isin(["At Risk", "Can't Lose"]).sum())
    kpi(c[2], "⚠️", "At risk / Can't lose", f"{n_risk:,}", "#E17055", "#FDCB6E")
    kpi(c[3], "💤", "Lost", f"{(rfm['Segment'] == 'Lost / Hibernating').sum():,}", "#636e72", "#b2bec3")
    a, b = st.columns(2)
    with a:
        show(px.pie(seg_stats, names="Segment", values="Customers", hole=0.5, title="Customers per segment"), 400)
    with b:
        show(px.bar(seg_stats.sort_values("Revenue"), x="Revenue", y="Segment", orientation="h",
                    color="Segment", title="Revenue per segment"), 400)
    sample = rfm.sample(min(3000, len(rfm)), random_state=0)
    fig = px.scatter(sample, x="Recency", y="Frequency", size="Monetary", color="Segment", log_y=True,
                     size_max=30, opacity=0.7, hover_data=["CustomerID", "Monetary"],
                     title="Recency vs Frequency (bubble = monetary)")
    show(fig, 480)
    st.dataframe(seg_stats.round(1), use_container_width=True)
    st.download_button("⬇️ Download RFM table (CSV)", rfm.to_csv(index=False).encode(), "rfm_segments.csv", "text/csv")

elif page == "🔁 Cohort Retention":
    hero("🔁 Cohort Retention", "Share of each monthly cohort that comes back")
    need_data()
    d = df_f[["CustomerID", "InvoiceDate"]].copy()
    d["OrderMonth"] = d["InvoiceDate"].dt.to_period("M")
    d["Cohort"] = d.groupby("CustomerID")["OrderMonth"].transform("min")
    d["Index"] = (d["OrderMonth"].dt.year - d["Cohort"].dt.year) * 12 + (d["OrderMonth"].dt.month - d["Cohort"].dt.month)
    cohort = d.groupby(["Cohort", "Index"])["CustomerID"].nunique().unstack(fill_value=0)
    ret = cohort.divide(cohort[0], axis=0)
    ret.index = ret.index.astype(str)
    fig = px.imshow(ret, text_auto=".0%", aspect="auto", color_continuous_scale="YlGnBu",
                    labels=dict(x="Months since first purchase", y="Cohort", color="Retention"),
                    title="Monthly retention by cohort")
    show(fig, 520)
    st.caption("Tip: the rows with the strongest first months are your best acquisition periods.")

elif page == "🧩 ML Segmentation":
    hero("🧩 Customer Segmentation (K-Means)", "11 customer clusters learnt on the first 10 months")
    (_k, _p, _pm, _pc, _ps, basket, train, test, users, scaler, scaled, km, cscore) = get_pipeline()
    n1 = (users["count"] == 1).sum()
    c = st.columns(4)
    kpi(c[0], "👥", "Customers", f"{len(users):,}", "#6C5CE7", "#A29BFE")
    kpi(c[1], "🧩", "Clusters", f"{N_CUSTOMER_CLUSTERS}", "#00B894", "#55EFC4")
    kpi(c[2], "🎯", "Silhouette", f"{cscore:.3f}", "#FD79A8", "#FF7675")
    kpi(c[3], "1️⃣", "One-time buyers", f"{n1 / len(users) * 100:.0f}%", "#E17055", "#FDCB6E")

    a, b = st.columns(2)
    with a:
        cnt = users["cluster"].value_counts().sort_index().reset_index()
        cnt.columns = ["cluster", "customers"]
        cnt["cluster"] = cnt["cluster"].astype(str)
        show(px.bar(cnt, x="cluster", y="customers", color="cluster", title="Customers per cluster"), 380)
    with b:
        pca = PCA(n_components=3)
        comp = pca.fit_transform(scaled)
        pdf = pd.DataFrame(comp, columns=["PC1", "PC2", "PC3"])
        pdf["cluster"] = users["cluster"].astype(str).values
        fig = px.scatter_3d(pdf, x="PC1", y="PC2", z="PC3", color="cluster", opacity=0.7,
                            title=f"PCA 3D (variance explained {pca.explained_variance_ratio_.sum():.0%})")
        fig.update_traces(marker_size=3)
        show(fig, 380)

    prof = users.groupby("cluster")[LIST_COLS + ["sum"]].mean()
    prof["size"] = users.groupby("cluster").size()
    st.subheader("Cluster profiles")
    try:
        st.dataframe(prof.round(2).style.background_gradient(cmap="PuBuGn"), use_container_width=True)
    except ImportError:  # pandas needs a newer matplotlib for colour gradients
        st.dataframe(prof.round(2), use_container_width=True)

    st.subheader("Radar comparison")
    sel = st.multiselect("Clusters to compare", sorted(users["cluster"].unique()), default=[0, 1, 2])
    attrs = ["count", "mean", "sum", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4"]
    rng = users[attrs].max().replace(0, 1)
    fig = go.Figure()
    for i, cl in enumerate(sel):
        vals = (prof.loc[cl, attrs] / rng).values
        fig.add_trace(go.Scatterpolar(r=np.r_[vals, vals[0]], theta=attrs + [attrs[0]], fill="toself",
                                      name=f"Cluster {cl}", line_color=COLORS[i % len(COLORS)], opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), title="Morphotypes (normalised to max)")
    show(fig, 520)
    st.download_button("⬇️ Download customers with clusters (CSV)", users.to_csv(index=False).encode(),
                       "customer_clusters.csv", "text/csv")

elif page == "🤖 Classification":
    hero("🤖 Classifying customers from the first purchase", "Which model recognises the segment best?")
    (_k, _p, _pm, _pc, _ps, basket, train, test, users, scaler, scaled, km, cscore) = get_pipeline()
    models, results, (X_test, Y_test) = train_classifiers(users[FEATURE_COLS], users["cluster"])
    best = results.loc[results["Accuracy (%)"].idxmax()]
    c = st.columns(3)
    kpi(c[0], "🏆", "Best model", best["Classifier"], "#6C5CE7", "#A29BFE")
    kpi(c[1], "🎯", "Best accuracy", f"{best['Accuracy (%)']}%", "#00B894", "#55EFC4")
    kpi(c[2], "🧮", "Features", f"{len(FEATURE_COLS)}", "#FD79A8", "#FF7675")
    a, b = st.columns(2)
    with a:
        show(px.bar(results, x="Accuracy (%)", y="Classifier", orientation="h", color="Classifier",
                    range_x=[0, 100], text="Accuracy (%)", title="Accuracy on hold-out set"), 400)
    with b:
        name = st.selectbox("Confusion matrix for", list(models.keys()), index=len(models) - 1)
        cm = confusion_matrix(Y_test, models[name].predict(X_test))
        show(px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                       labels=dict(x="Predicted", y="True"), title=f"Confusion matrix: {name}"), 400)
    rf = models["Random Forest"].best_estimator_
    imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": rf.feature_importances_}).sort_values("importance")
    show(px.bar(imp, x="importance", y="feature", orientation="h", color="importance",
                color_continuous_scale="Magma", title="Random Forest feature importance"), 340)
    st.dataframe(results, use_container_width=True)

elif page == "🧪 Test (last 2 months)":
    hero("🧪 Testing on the last two months", "Does the model hold up on unseen time?")
    (_k, _p, _pm, _pc, _ps, basket, train, test, users, scaler, scaled, km, cscore) = get_pipeline()
    models, results, _ = train_classifiers(users[FEATURE_COLS], users["cluster"])
    if test.empty:
        st.warning("No test data after 2011-10-01.")
        st.stop()
        raise SystemExit
    tu = per_user(test, first_last=False)
    tu["count"] = 5 * tu["count"]
    tu["sum"] = tu["count"] * tu["mean"]
    Y = km.predict(scaler.transform(tu[LIST_COLS].to_numpy()))
    X = tu[FEATURE_COLS]
    res = pd.DataFrame([{"Classifier": n, "Accuracy (%)": round(100 * metrics.accuracy_score(Y, m.predict(X)), 2)}
                        for n, m in models.items()])
    st.metric("Test customers", f"{len(tu):,}")
    show(px.bar(res, x="Classifier", y="Accuracy (%)", color="Classifier", range_y=[0, 100],
                text="Accuracy (%)", title="Accuracy on the last 2 months"), 420)
    st.caption("Seasonality (e.g. Christmas) is a known bias: categories learnt on 10 months may differ "
               "from those inferred on the last 2.")

elif page == "🔮 Predict a Customer":
    hero("🔮 Predict the segment of a new customer", "Enter the first basket and get the cluster")
    (_k, _p, _pm, _pc, _ps, basket, train, test, users, scaler, scaled, km, cscore) = get_pipeline()
    models, results, _ = train_classifiers(users[FEATURE_COLS], users["cluster"])
    model_name = st.selectbox("Model", list(models.keys()), index=len(models) - 1)
    mean_basket = st.number_input("First basket amount (£)", min_value=0.0, value=300.0, step=10.0)
    st.write("Share of the basket in each product category (normalised to 100%)")
    cols = st.columns(N_PRODUCT_CLUSTERS)
    shares = [cols[i].number_input(f"categ_{i} (%)", 0.0, 100.0, 20.0, 5.0) for i in range(N_PRODUCT_CLUSTERS)]
    if st.button("🔮 Predict", type="primary"):
        total = sum(shares)
        if total == 0:
            st.error("At least one category share must be > 0.")
        else:
            shares = [s / total * 100 for s in shares]
            x = pd.DataFrame([[mean_basket] + shares], columns=FEATURE_COLS)
            model = models[model_name]
            pred = int(model.predict(x)[0])
            c = st.columns(3)
            kpi(c[0], "🎯", "Predicted cluster", f"{pred}", "#6C5CE7", "#FD79A8")
            sub = users[users["cluster"] == pred]
            kpi(c[1], "👥", "Customers in cluster", f"{len(sub):,}", "#00B894", "#55EFC4")
            kpi(c[2], "💰", "Avg total spend", f"£{sub['sum'].mean():,.0f}", "#E17055", "#FDCB6E")
            try:
                proba = model.predict_proba(x)[0]
                pdf = pd.DataFrame({"cluster": [str(c_) for c_ in model.classes_], "probability": proba})
                show(px.bar(pdf, x="cluster", y="probability", color="cluster", title="Class probabilities"), 340)
            except Exception:
                pass
            st.dataframe(sub[LIST_COLS + ["sum"]].mean().round(2).to_frame("cluster average"),
                         use_container_width=True)

elif page == "🔎 Customer Lookup":
    hero("🔎 Customer Lookup", "Drill into a single customer")
    need_data()
    ids = sorted(df_f["CustomerID"].unique())
    cid = st.selectbox("Customer ID", ids)
    cd = df_f[df_f["CustomerID"] == cid]
    c = st.columns(4)
    kpi(c[0], "💰", "Total spend", f"£{cd['TotalPrice'].sum():,.0f}", "#6C5CE7", "#A29BFE")
    kpi(c[1], "🧾", "Orders", f"{cd['InvoiceNo'].nunique()}", "#00B894", "#55EFC4")
    kpi(c[2], "📦", "Distinct products", f"{cd['StockCode'].nunique()}", "#FD79A8", "#FF7675")
    kpi(c[3], "🌍", "Country", f"{cd['Country'].iloc[0]}", "#0984E3", "#74B9FF")
    a, b = st.columns(2)
    with a:
        m = cd.groupby("Month")["TotalPrice"].sum().reset_index()
        show(px.bar(m, x="Month", y="TotalPrice", color="TotalPrice", title="Monthly spend"), 340)
    with b:
        tp = cd.groupby("Description")["TotalPrice"].sum().nlargest(10).reset_index()
        fig = px.bar(tp, x="TotalPrice", y="Description", orientation="h", title="Top products")
        fig.update_yaxes(autorange="reversed")
        show(fig, 340)
    st.dataframe(cd[["InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "TotalPrice"]]
                 .sort_values("InvoiceDate"), use_container_width=True)

elif page == "📋 Data Explorer":
    hero("📋 Data Explorer", "Browse, check quality and export")
    t1, t2 = st.tabs(["Filtered sales", "Data quality"])
    with t1:
        need_data()
        st.write(f"{len(df_f):,} rows after filters")
        st.dataframe(df_f.drop(columns=["Month", "Weekday", "Hour"]).head(1000), use_container_width=True)
        st.download_button("⬇️ Download filtered sales (CSV)", df_f.to_csv(index=False).encode(),
                           "filtered_sales.csv", "text/csv")
    with t2:
        c = st.columns(3)
        kpi(c[0], "📄", "Raw rows", f"{len(df_raw):,}", "#6C5CE7", "#A29BFE")
        kpi(c[1], "🚫", "No CustomerID", f"{df_raw['CustomerID'].isna().sum():,}", "#E17055", "#FDCB6E")
        kpi(c[2], "✅", "After cleaning", f"{len(df_cleaned):,}", "#00B894", "#55EFC4")
        info = pd.DataFrame({"type": df_raw.dtypes.astype(str), "null (nb)": df_raw.isnull().sum(),
                             "null (%)": (df_raw.isnull().sum() / len(df_raw) * 100).round(2)})
        a, b = st.columns(2)
        with a:
            st.dataframe(info, use_container_width=True)
        with b:
            show(px.bar(df_raw.isnull().sum().reset_index(name="nulls"), x="index", y="nulls", color="index",
                        title="Missing values per column"), 300)
        st.write(f"Cancellations matched & removed: **{clean_info['entry_to_remove']}** • "
                 f"without prior order removed: **{clean_info['doubtful_entry']}**")
        num_cols = df_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
        corr = df_raw[num_cols].corr()
        show(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation heatmap"), 380)