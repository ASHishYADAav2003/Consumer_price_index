import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="CPI Analysis & Forecasting", layout="wide")
st.title("📊 Consumer Price Index (CPI) Analysis & Forecasting")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_consumer_price_index.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
num_cols = df.select_dtypes(include=["float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

le_sector = LabelEncoder()
le_month = LabelEncoder()

df["Sector_encoded"] = le_sector.fit_transform(df["Sector"])
df["Month_encoded"] = le_month.fit_transform(df["Month"])


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["EDA", "ML Models", "Future Forecasting"]
)


# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------
if section == "EDA":
    st.header("📈 Exploratory Data Analysis")

    # Line Plot
    st.subheader("General CPI Trend by Sector")
    fig, ax = plt.subplots()
    for sector in df["Sector"].unique():
        temp = df[df["Sector"] == sector]
        ax.plot(temp["Year"], temp["General index"], label=sector)
    ax.set_xlabel("Year")
    ax.set_ylabel("General Index")
    ax.legend()
    st.pyplot(fig)

    # Histogram
    st.subheader("Distribution of General CPI Index")
    fig, ax = plt.subplots()
    ax.hist(df["General index"])
    ax.set_xlabel("General Index")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("Sector Distribution")
    sector_counts = df["Sector"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sector_counts, labels=sector_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

    # Boxplot by Sector
    st.subheader("CPI by Sector")
    fig, ax = plt.subplots()
    df.boxplot(column="General index", by="Sector", ax=ax)
    plt.suptitle("")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr)
    fig.colorbar(im)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)


# --------------------------------------------------
# ML MODELS SECTION
# --------------------------------------------------
if section == "ML Models":
    st.header("🤖 Machine Learning Models")

    X = df[["Year", "Sector_encoded", "Month_encoded"]]
    y = df["General index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    results = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "R2 Score": [
            r2_score(y_test, lr_pred),
            r2_score(y_test, rf_pred),
            r2_score(y_test, xgb_pred)
        ],
        "MSE": [
            mean_squared_error(y_test, lr_pred),
            mean_squared_error(y_test, rf_pred),
            mean_squared_error(y_test, xgb_pred)
        ]
    }

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Model Comparison Plot
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["R2 Score"])
    ax.set_ylabel("R2 Score")
    ax.set_title("Model Comparison")
    st.pyplot(fig)


# --------------------------------------------------
# FUTURE FORECASTING SECTION
# --------------------------------------------------
if section == "Future Forecasting":
    st.header("🔮 Future CPI Forecasting (2026–2030)")

    X = df[["Year", "Sector_encoded", "Month_encoded"]]
    y = df["General index"]

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    xgb.fit(X, y)

    future_years = range(2026, 2031)
    future_data = []

    for year in future_years:
        for sector in df["Sector_encoded"].unique():
            for month in range(12):
                future_data.append([year, sector, month])

    future_df = pd.DataFrame(
        future_data,
        columns=["Year", "Sector_encoded", "Month_encoded"]
    )

    future_df["Predicted_General_Index"] = xgb.predict(future_df)

    forecast_summary = future_df.groupby("Year")[
        "Predicted_General_Index"
    ].mean()

    st.subheader("Forecasted CPI Values")
    st.dataframe(forecast_summary)

    fig, ax = plt.subplots()
    ax.plot(
        forecast_summary.index,
        forecast_summary.values,
        marker="o"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted General CPI Index")
    ax.set_title("Future CPI Forecast")
    st.pyplot(fig)
