# 🛍️ Customer Segmentation — E-commerce Analytics & ML Dashboard

> Segment online-retail customers with **K-Means**, then predict a new customer's segment **from their very first purchase** — explored in a Jupyter Notebook and delivered as an interactive **Streamlit dashboard**.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Streamlit Dashboard](#-streamlit-dashboard)
8. [Installation & Usage](#-installation--usage)
9. [Tech Stack](#-tech-stack)
10. [Limitations & Future Work](#-limitations--future-work)
11. [Acknowledgements](#-acknowledgements)
12. [Author](#-author)

---

## 🔎 Overview

Online retailers collect thousands of transactions, but raw invoices don't tell you **who your customers are**. This project turns a year of e-commerce purchase data into actionable customer segments and answers a practical business question:

> *"A customer just made their first purchase. What kind of customer are they likely to become?"*

The workflow has two parts:

| Part | File | Purpose |
|------|------|---------|
| 📓 **Analysis notebook** | `customer-segmentation.ipynb` | End-to-end exploration: cleaning, EDA, product clustering, customer clustering, classifier training and time-based testing |
| 🖥️ **Dashboard app** | `streamlit_app.py` | Interactive multi-page app to explore sales, RFM segments, cohorts, ML clusters, model accuracy and live predictions |

---

## ✨ Key Features

- **Robust data cleaning** — removes records without `CustomerID`, drops duplicates, and *matches cancellations to the original orders* (adds a `QuantityCanceled` field instead of blindly deleting rows).
- **Product categorisation** — extracts keywords from product descriptions and groups products into **5 categories** using K-Means on keyword + price-range features.
- **Customer segmentation** — clusters customers into **11 behavioural segments** based on order count, basket size and spend share per product category.
- **Classification from the first purchase** — 6+ classifiers (SVM, Logistic Regression, kNN, Decision Tree, Random Forest, Gradient Boosting) tuned with `GridSearchCV`, plus a **soft-voting ensemble**.
- **Time-aware validation** — models are trained on the first 10 months and tested on the **last 2 months** of unseen data.
- **RFM analysis & cohort retention** in the dashboard for classic marketing-style segmentation.
- **Interactive visuals** — choropleth maps, treemaps, heatmaps, radar charts, 3D PCA scatter and word clouds.
- **"Predict a Customer" tool** — enter a first basket and get the predicted segment with class probabilities.
- **Export-ready** — download RFM tables, cluster assignments and filtered sales as CSV.

---

## 📦 Dataset

The project uses the **Online Retail** transactions dataset (a UK-based online store selling gifts; ~541k rows covering **01 Dec 2010 → 09 Dec 2011**).

| Column | Description |
|--------|-------------|
| `InvoiceNo` | Invoice number. Starts with **`C`** if the order was cancelled |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units per transaction (negative for cancellations) |
| `InvoiceDate` | Date and time of the transaction |
| `UnitPrice` | Price per unit (£) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

**Quick facts from the data-prep stage**

| Metric | Value |
|--------|-------|
| Raw rows | 541,909 |
| Rows missing `CustomerID` (removed) | 135,080 (~24.9%) |
| Rows after dropping missing IDs | 406,829 |
| Countries | 37 |
| Cancelled share of transactions | ~16% |
| Cancellations matched to an earlier order | 7,521 |
| "Doubtful" cancellations (no prior purchase found) | 1,226 |
| Customers with baskets used for modelling | 4,327 |
| One-time buyers | 1,489 (≈34%) |

> ⚠️ **The dataset (`data.csv`) is not included in this repository.** Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) or [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data) and place it in the project root as `data.csv` (encoding: `ISO-8859-1`).

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── customer-segmentation.ipynb   # Full analysis & modelling notebook
├── streamlit_app.py              # Interactive dashboard
├── data.csv                      # Dataset (download separately — not tracked)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🧠 Methodology

### 1️⃣ Data Preparation
- Load `data.csv` with `ISO-8859-1` encoding; parse `InvoiceDate` as datetime.
- Drop rows with no `CustomerID` (they can't be attributed to a customer) and remove exact duplicates.

### 2️⃣ Exploratory Analysis
- **Countries:** orders per country shown on a choropleth map — the UK dominates.
- **Cancellations:** for every negative-quantity row (excluding `Discount`), the code looks for an *earlier* purchase by the same customer of the same `StockCode` with enough quantity. Matched cancellations are recorded in `QuantityCanceled` and the cancellation row is removed; unmatched ones are flagged as doubtful and dropped.
- **Special stock codes:** non-product entries (postage, bank charges, discounts, etc.) are identified and examined.
- **Basket price:** `TotalPrice = UnitPrice × (Quantity − QuantityCanceled)`, aggregated per invoice and visualised by price range.

### 3️⃣ Product Categories
- Extract meaningful **keywords** from descriptions (noun extraction with NLP in the notebook; regex-based extraction in the app), ignoring colours and generic words.
- Build a **one-hot matrix** (product × keyword) and add **6 price-range columns** (`0–1`, `1–2`, `2–3`, `3–5`, `5–10`, `>10` £), which produces more balanced groups.
- Cluster with **K-Means (k = 5)**, selecting runs using the **silhouette score**.
- Interpret clusters with **word clouds**, silhouette plots and **PCA** visualisation.

### 4️⃣ Customer Categories
- Add `categ_0 … categ_4` — the amount each customer spends in each product cluster.
- Aggregate orders into baskets and **split the data in time**: first 10 months for training, everything from **2011-10-01** for testing.
- Build per-customer features: number of orders, min / max / mean basket, and the **% of spend in each product category**.
- Standardise (`StandardScaler`) and cluster with **K-Means (k = 11)** → customer "morphotypes".

### 5️⃣ Classification
Goal: predict the customer's cluster using only information available at the **first purchase** — the basket amount (`mean`) and the percentage spent in each of the 5 product categories (`categ_0 … categ_4`).

- Train/test split (80/20) with 5-fold `GridSearchCV` hyper-parameter tuning.
- Models: **Linear SVM, Logistic Regression, k-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting** (AdaBoost is also explored in the notebook).
- A **soft-voting ensemble** of Random Forest + Gradient Boosting + kNN combines the strongest models.

### 6️⃣ Testing on Unseen Time
- The last two months are re-processed with the same pipeline (counts and sums re-weighted to match the training period length).
- Predictions are compared against the clusters inferred from the test period's full behaviour.

---

## 📊 Results

### Hold-out accuracy (first 10 months, 80/20 split)

| Classifier | Accuracy |
|------------|:--------:|
| Support Vector Machine | 81.52% |
| Decision Tree | 82.22% |
| k-Nearest Neighbors | 83.72% |
| Logistic Regression | 84.41% |
| AdaBoost | 84.87% |
| Random Forest | 86.61% |
| Gradient Boosting | 88.57% |
| **Voting (RF + GB + kNN)** | **≈88%** |

### Accuracy on the last 2 months (unseen time period)

| Classifier | Accuracy |
|------------|:--------:|
| Support Vector Machine | 69.07% |
| Logistic Regression | 68.87% |
| k-Nearest Neighbors | 69.15% |
| Decision Tree | 73.38% |
| Gradient Boosting | 75.23% |
| Random Forest | 75.97% |
| **Voting ensemble** | **75.77%** |

### Takeaways
- **Tree-based and ensemble models** perform best, with Random Forest and Gradient Boosting leading.
- Accuracy drops on the final two months. **Seasonality** (the Christmas period) is a known source of bias: the product mix and spending of Nov–Dec differ from the 10 months used to learn the segments.
- The customer clustering silhouette score is **≈ 0.31** in the notebook and **≈ 0.22** in the dashboard (which clusters only the 3,608 customers active in the first 10 months), indicating moderately overlapping but usable segments.
- Roughly **one third to 40% of customers buy only once** (34% in the notebook, 40% in the dashboard's training window), which makes early identification of high-value customers especially valuable.

> 📝 *Exact figures can vary slightly between runs and library versions (random initialisation, grid-search folds). The numbers above come from the notebook outputs.*

---

## 🖥️ Streamlit Dashboard

The app (`streamlit_app.py`) provides **11 pages** accessible from the sidebar, plus global **country** and **date-range** filters.

| Page | What you can do |
|------|-----------------|
| 🏠 **Dashboard** | KPIs (revenue, orders, customers, AOV, products, countries), monthly revenue/orders, top countries & products, weekday × hour heatmap, basket-value distribution |
| 🌍 **Geography** | Choropleth map (log-scaled) for revenue / orders / customers, top-20 treemap, country table |
| 📦 **Products** | Best sellers by revenue or quantity, price-vs-quantity bubble chart, keyword frequency, product clusters + PCA, per-cluster word clouds |
| 💎 **RFM Segments** | Recency / Frequency / Monetary scoring into *Champions, Loyal, New/Promising, Can't Lose, At Risk, Lost/Hibernating, Needs Attention*; CSV download |
| 🔁 **Cohort Retention** | Monthly cohort retention heatmap |
| 🧩 **ML Segmentation** | 11 K-Means clusters, 3D PCA, cluster profiles, radar comparison, CSV export |
| 🤖 **Classification** | Grid-searched classifiers, accuracy comparison, confusion matrices, Random Forest feature importance |
| 🧪 **Test (last 2 months)** | Evaluate every model on the unseen final two months |
| 🔮 **Predict a Customer** | Enter first-basket amount + category shares → predicted cluster, probabilities and cluster profile |
| 🔎 **Customer Lookup** | Drill into a single customer's spend, orders and top products |
| 📋 **Data Explorer** | Browse filtered data, data-quality report, correlation heatmap, CSV export |

**Performance notes:** data loading, cleaning, clustering and model training are cached with `st.cache_data` / `st.cache_resource`, so heavy steps run only once per session.

> 💡 **Tip:** The look of the app (background, text and heading colours) can be changed by editing the `MAIN_BG`, `TEXT_COLOR` and `HEADING_COLOR` variables near the top of `streamlit_app.py`.

### 📸 Screenshots

| Sales Dashboard | RFM Segments |
|-----------------|--------------|
| ![Sales Dashboard](screenshots/Dashboard.png) | ![RFM Segments](screenshots/RFM_Segments.png) |

| ML Segmentation (K-Means) | Predict a Customer |
|---------------------------|--------------------|
| ![ML Segmentation](screenshots/ML_segmentation.png) | ![Predict a Customer](screenshots/Predict_a_Customer.png) |

---

## 🚀 Installation & Usage

### Prerequisites
- Python **3.9+**
- `pip` (and optionally a virtual environment)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-segmentation.git
cd customer-segmentation
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt</b></summary>

```text
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
wordcloud
nltk
spacy
jupyter
```
</details>

For the notebook's keyword-extraction step, also download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

### 4. Add the dataset
Place `data.csv` in the project root (see [Dataset](#-dataset)).

### 5a. Run the Streamlit dashboard
```bash
streamlit run streamlit_app.py
```
Then open the URL shown in the terminal (usually `http://localhost:8501`). You can either **upload `data.csv` in the sidebar** or let the app read `data.csv` from the project folder.

### 5b. Run the notebook
```bash
jupyter notebook customer-segmentation.ipynb
```
Run the cells from top to bottom.

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data handling | Pandas, NumPy |
| Machine learning | scikit-learn (KMeans, PCA, GridSearchCV, SVM, Logistic Regression, kNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, VotingClassifier) |
| NLP | NLTK, spaCy, WordCloud |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App framework | Streamlit |
| Environment | Jupyter Notebook |

---

## ⚠️ Limitations & Future Work

- **Seasonality bias** — 10 months of training data can't capture a full year; accuracy falls on the Nov–Dec test window. More than one year of data would help.
- **Weak product-cluster separation** — keyword-based clusters overlap; embeddings (e.g. TF-IDF / sentence embeddings) could improve them.
- **Features at first purchase are limited** — adding country, time of purchase or product-level details may boost prediction.
- **Fixed number of clusters** — k = 5 (products) and k = 11 (customers) are chosen for this dataset; the app does not yet let users change them.

**Ideas for the roadmap**
- [ ] Customer Lifetime Value (CLV) prediction
- [ ] Market-basket analysis / recommendation engine
- [ ] Churn prediction using recency features
- [ ] Hyper-parameter / cluster-count controls in the UI
- [ ] Deploy on Streamlit Community Cloud

---

## 🙏 Acknowledgements

- Dataset: *Online Retail* — Daqing Chen et al., UCI Machine Learning Repository.
- The analysis structure (cancellation handling, product categories → customer categories → classifiers) is adapted from the public Kaggle notebook **"Customer Segmentation"** by *Fabien Daniel*, extended here with an RFM module, cohort retention, extra visualisations, time-based testing and a full Streamlit app.

---

## 👤 Author

**Gaurav Rajpoot**
B.Tech (CSE – AI/ML), IMS Engineering College, Ghaziabad

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/your-linkedin)
- Email: your.email@example.com

---

⭐ If you found this project useful, consider giving it a star!
