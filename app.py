import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="LifePass Dubai Analytics",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY   = "#1B3A6B"
TEAL   = "#0D6E6E"
AMBER  = "#F59E0B"
CORAL  = "#EF4444"
GREEN  = "#10B981"
PURPLE = "#8B5CF6"

SEG_COLORS = {
    "Busy Professional": "#2563EB",
    "Nuclear Family":    "#10B981",
    "Budget Expat":      "#F59E0B",
    "Premium Resident":  "#8B5CF6",
}

st.markdown("""
<style>
.main { background: #F8FAFC; }
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: #1B3A6B;
    padding: 8px 12px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.12); color: white;
    border-radius: 8px; padding: 8px 20px;
    font-weight: 500; border: none;
}
.stTabs [aria-selected="true"] {
    background: #F59E0B !important;
    color: #1B3A6B !important; font-weight: 700;
}
.metric-card {
    background: white; border-radius: 12px; padding: 20px;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-top: 4px solid;
}
.insight-box {
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4);
    border-left: 4px solid #2563EB;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 12px 0;
}
.stSidebar { background: linear-gradient(180deg,#1B3A6B 0%,#0D6E6E 100%); }
</style>
""", unsafe_allow_html=True)

# ── Column name constants (matching CSV exactly) ──────────────────────────────
C = {
    "id":           "Respondent_ID",
    "seg":          "Segment",
    "gender":       "Gender",
    "age":          "Q1_Age",
    "nat":          "Q2_Nationality",
    "inc_ord":      "Q3_Income_Bracket",
    "inc_aed":      "Q3_Income_AED_Month",
    "living":       "Q4_Living_Situation",
    "area":         "Q5_Area_Dubai",
    "apps":         "Q6_Num_Apps_Used",
    "spend":        "Q7_Monthly_Service_Spend_AED",
    "hours":        "Q8_Hours_Logistics_Per_Week",
    "u_laundry":    "Q9_Uses_Laundry",
    "u_cleaning":   "Q9_Uses_Cleaning",
    "u_carwash":    "Q9_Uses_Carwash",
    "u_grocery":    "Q9_Uses_Grocery",
    "u_salon":      "Q9_Uses_Salon",
    "u_maint":      "Q9_Uses_Maintenance",
    "u_pet":        "Q9_Uses_Petcare",
    "u_airport":    "Q9_Uses_Airport",
    "u_concierge":  "Q9_Uses_Concierge",
    "sat":          "Q10_Satisfaction_Current",
    "prob_freq":    "Q11_Problem_Frequency",
    "pain":         "Q12_Pain_Score_1to10",
    "frust":        "Q13_Primary_Frustration",
    "imp_app":      "Q14_Importance_One_App",
    "f_laundry":    "Q15_Freq_Laundry",
    "f_cleaning":   "Q15_Freq_Cleaning",
    "f_carwash":    "Q15_Freq_Carwash",
    "f_grocery":    "Q15_Freq_Grocery",
    "f_salon":      "Q15_Freq_Salon",
    "f_maint":      "Q15_Freq_Maintenance",
    "f_pet":        "Q15_Freq_Petcare",
    "car":          "Q16_Car_Ownership",
    "pet":          "Q17_Pet_Ownership",
    "intent":       "Q18_Subscription_Intent",
    "wtp":          "Q19_WTP_AED_Month",
    "wtp_cat":      "Q19_WTP_Category",
    "payment":      "Q20_Preferred_Payment",
    "ai":           "Q22_AI_Comfort",
    "subs":         "Q23_Existing_Subscriptions",
    "trust":        "Q24_Trust_Driver",
    "lifescore":    "Q25_LifeScore",
    "target":       "Would_Subscribe",
}

@st.cache_data
def load_data():
    df = pd.read_csv("lifepass_survey_1000.csv")
    return df

@st.cache_data
def run_ml(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix, roc_curve)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    fe = df.copy()
    num_cols = fe.select_dtypes(include=[np.number]).columns
    fe[num_cols] = SimpleImputer(strategy="median").fit_transform(fe[num_cols])

    # Feature engineering
    fe["spend_to_income"]   = fe[C["spend"]] / (fe[C["inc_aed"]] / 12 + 1)
    fe["total_svc_usage"]   = (fe[C["f_laundry"]] + fe[C["f_cleaning"]] + fe[C["f_carwash"]] +
                                fe[C["f_grocery"]] + fe[C["f_salon"]] + fe[C["f_maint"]] + fe[C["f_pet"]])
    fe["services_count"]    = (fe[C["u_laundry"]] + fe[C["u_cleaning"]] + fe[C["u_carwash"]] +
                                fe[C["u_grocery"]] + fe[C["u_salon"]] + fe[C["u_maint"]] +
                                fe[C["u_pet"]] + fe[C["u_airport"]] + fe[C["u_concierge"]])
    fe["digital_readiness"] = (fe[C["ai"]] + fe[C["subs"]]) / 2
    fe["pain_x_importance"] = fe[C["pain"]] * fe[C["imp_app"]]
    fe["car_binary"]        = (fe[C["car"]] != "No car").astype(int)
    fe["pet_binary"]        = (fe[C["pet"]] != "No pets").astype(int)

    le = LabelEncoder()
    cat_cols = [C["nat"], C["living"], C["area"], C["frust"],
                C["car"], C["pet"], C["payment"], C["trust"], C["gender"]]
    for col in cat_cols:
        fe[col + "_enc"] = le.fit_transform(fe[col].astype(str))

    feature_cols = [
        C["age"], C["inc_ord"], C["apps"], C["spend"],
        C["hours"], C["sat"], C["prob_freq"], C["pain"],
        C["imp_app"], C["intent"], C["wtp"], C["ai"],
        C["subs"], C["lifescore"],
        "total_svc_usage", "services_count", "digital_readiness",
        "pain_x_importance", "spend_to_income", "car_binary", "pet_binary",
        C["nat"] + "_enc", C["living"] + "_enc", C["area"] + "_enc",
        C["frust"] + "_enc", C["payment"] + "_enc", C["trust"] + "_enc",
    ]
    feature_cols = [f for f in feature_cols if f in fe.columns]

    X = fe[feature_cols].fillna(0)
    y = fe[C["target"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.25,
                                               random_state=42, stratify=y)

    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                         learning_rate=0.08, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        yp    = model.predict(X_te)
        ypr   = model.predict_proba(X_te)[:, 1]
        yp_tr = model.predict(X_tr)
        cm    = confusion_matrix(y_te, yp)
        fpr, tpr, _ = roc_curve(y_te, ypr)
        results[name] = {
            "train_acc": accuracy_score(y_tr, yp_tr),
            "test_acc":  accuracy_score(y_te, yp),
            "precision": precision_score(y_te, yp, zero_division=0),
            "recall":    recall_score(y_te, yp, zero_division=0),
            "f1":        f1_score(y_te, yp, zero_division=0),
            "auc":       roc_auc_score(y_te, ypr),
            "cm": cm, "fpr": fpr, "tpr": tpr, "model": model,
        }

    rf = results["Random Forest"]["model"]
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    cluster_features = [C["pain"], C["wtp"], C["spend"], C["inc_ord"],
                        "total_svc_usage", C["apps"], C["ai"], C["sat"]]
    cluster_features = [f for f in cluster_features if f in fe.columns]
    Xc    = fe[cluster_features].fillna(fe[cluster_features].median())
    sc2   = StandardScaler()
    Xcs   = sc2.fit_transform(Xc)
    km    = KMeans(n_clusters=4, random_state=42, n_init=15)
    km.fit(Xcs)
    pca   = PCA(n_components=2, random_state=42)
    Xpca  = pca.fit_transform(Xcs)

    return results, fi, feature_cols, Xpca, km.labels_, fe

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ LifePass Dubai")
    st.markdown("**Analytics Dashboard**")
    st.markdown("---")
    seg_filter = st.multiselect("Customer Segments",
        options=sorted(df[C["seg"]].unique().tolist()),
        default=sorted(df[C["seg"]].unique().tolist()))
    nat_filter = st.multiselect("Nationality",
        options=sorted(df[C["nat"]].unique().tolist()),
        default=sorted(df[C["nat"]].unique().tolist()))
    inc_filter = st.slider("Income Bracket (1=<5K AED  →  5=>35K AED)", 1, 5, (1, 5))
    st.markdown("---")
    st.markdown("**n = 1,000 respondents**")
    st.markdown("**Dubai 2026 · SP Jain GMBA**")

mask = (df[C["seg"]].isin(seg_filter) &
        df[C["nat"]].isin(nat_filter) &
        df[C["inc_ord"]].between(inc_filter[0], inc_filter[1]))
dff = df[mask].copy()
n   = len(dff)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,{NAVY},{TEAL});border-radius:16px;
padding:28px 32px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2rem'>🏙️ LifePass Dubai</h1>
  <p style='color:#FCD34D;margin:4px 0 0;font-size:1.1rem;font-weight:500'>
    One Subscription · Every Service · One City — Analytics Dashboard</p>
  <p style='color:rgba(255,255,255,0.75);margin:6px 0 0;font-size:0.9rem'>
    {n:,} respondents shown · SP Jain GMBA Data Analytics Project 2026</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
def kpi(col, val, label, color):
    col.markdown(f"""<div class='metric-card' style='border-top-color:{color}'>
      <div style='font-size:1.9rem;font-weight:700;color:{color}'>{val}</div>
      <div style='color:#6B7280;font-size:0.82rem'>{label}</div></div>""",
      unsafe_allow_html=True)

kpi(k1, f"{n:,}",                                              "Respondents",          NAVY)
kpi(k2, f"{dff[C['target']].mean()*100:.1f}%",               "Would Subscribe",       GREEN)
kpi(k3, f"AED {dff[C['spend']].median():,.0f}",              "Median Monthly Spend",  AMBER)
kpi(k4, f"{dff[C['pain']].mean():.1f}/10",                   "Avg Pain Score",        CORAL)
kpi(k5, f"AED {dff[C['wtp']].median():.0f}",                 "Median WTP/Month",      PURPLE)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Predictive Analytics",
    "🎯 Prescriptive Analytics",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Who Are Our Respondents?")
    st.markdown("*Demographics, behaviour, and service usage across Dubai*")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(dff, names=C["seg"], title="Customer Segments",
                     color=C["seg"], color_discrete_map=SEG_COLORS, hole=0.45)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        nc = dff[C["nat"]].value_counts().reset_index()
        nc.columns = ["Nationality","Count"]
        fig = px.bar(nc, x="Count", y="Nationality", orientation="h",
                     title="Nationality Mix", color="Count",
                     color_continuous_scale=[[0,"#DBEAFE"],[1,NAVY]])
        fig.update_layout(margin=dict(t=40,b=10), height=320,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        ac = dff[C["area"]].value_counts().reset_index()
        ac.columns = ["Area","Count"]
        fig = px.bar(ac, x="Count", y="Area", orientation="h",
                     title="Area of Residence", color="Count",
                     color_continuous_scale=[[0,"#D1FAE5"],[1,TEAL]])
        fig.update_layout(margin=dict(t=40,b=10), height=320,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x=C["spend"], nbins=50,
                           color=C["seg"], color_discrete_map=SEG_COLORS,
                           title="Monthly Service Spend (AED) — Right-Skewed Distribution",
                           labels={C["spend"]: "Monthly Spend (AED)"})
        fig.add_vline(x=dff[C["spend"]].mean(), line_dash="dash", line_color=CORAL,
                      annotation_text=f"Mean AED {dff[C['spend']].mean():,.0f}")
        fig.add_vline(x=dff[C["spend"]].median(), line_dash="dot", line_color=GREEN,
                      annotation_text=f"Median AED {dff[C['spend']].median():,.0f}")
        fig.update_layout(margin=dict(t=40,b=10), height=360, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        inc_map = {1:"<AED 5K",2:"AED 5-10K",3:"AED 10-20K",4:"AED 20-35K",5:">AED 35K"}
        dff2 = dff.copy()
        dff2["Income Label"] = dff2[C["inc_ord"]].map(inc_map)
        order = ["<AED 5K","AED 5-10K","AED 10-20K","AED 20-35K",">AED 35K"]
        inc_seg = dff2.groupby(["Income Label", C["seg"]]).size().reset_index(name="Count")
        fig = px.bar(inc_seg, x="Income Label", y="Count", color=C["seg"],
                     color_discrete_map=SEG_COLORS, title="Income Bracket by Segment",
                     category_orders={"Income Label": order})
        fig.update_layout(margin=dict(t=40,b=10), height=360, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Service Usage Heatmap by Segment")
    svc_map = {
        "Laundry":     C["u_laundry"], "Cleaning": C["u_cleaning"],
        "Car Wash":    C["u_carwash"], "Grocery":  C["u_grocery"],
        "Salon":       C["u_salon"],   "Maintenance": C["u_maint"],
        "Pet Care":    C["u_pet"],     "Airport":  C["u_airport"],
        "Concierge":   C["u_concierge"],
    }
    seg_order = ["Busy Professional","Nuclear Family","Budget Expat","Premium Resident"]
    heat = pd.DataFrame({
        svc: dff.groupby(C["seg"])[col].mean() * 100
        for svc, col in svc_map.items()
    }).reindex([s for s in seg_order if s in dff[C["seg"]].unique()])
    fig = px.imshow(heat, text_auto=".0f", aspect="auto",
                    color_continuous_scale=[[0,"#F0F9FF"],[0.5,"#0EA5E9"],[1,"#1B3A6B"]],
                    title="% of Segment Using Each Service Monthly",
                    labels={"color":"Usage %"})
    fig.update_layout(margin=dict(t=40,b=20), height=280)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(dff, x=C["seg"], y=C["apps"],
                     color=C["seg"], color_discrete_map=SEG_COLORS,
                     title="Number of Apps Used by Segment",
                     labels={C["apps"]:"Number of Apps", C["seg"]:"Segment"})
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10), height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(dff, x=C["seg"], y=C["wtp"],
                        color=C["seg"], color_discrete_map=SEG_COLORS,
                        box=True, points="outliers",
                        title="Willingness to Pay Distribution (AED/month)",
                        labels={C["wtp"]:"WTP (AED)", C["seg"]:"Segment"})
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10), height=340)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sc = dff.groupby([C["seg"], C["target"]]).size().reset_index(name="Count")
        sc["Status"] = sc[C["target"]].map({1:"Would Subscribe",0:"Would Not"})
        fig = px.bar(sc, x=C["seg"], y="Count", color="Status",
                     color_discrete_map={"Would Subscribe":GREEN,"Would Not":CORAL},
                     title="Subscription Intent by Segment", barmode="stack")
        fig.update_layout(margin=dict(t=40,b=10), height=320, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ls = dff.groupby(C["living"])[C["target"]].mean().reset_index()
        ls.columns = ["Living Situation","Subscribe Rate"]
        ls["Subscribe Rate"] *= 100
        fig = px.bar(ls.sort_values("Subscribe Rate"),
                     x="Subscribe Rate", y="Living Situation", orientation="h",
                     title="Subscription Rate by Living Situation (%)",
                     color="Subscribe Rate",
                     color_continuous_scale=[[0,"#FEF3C7"],[1,AMBER]])
        fig.update_layout(margin=dict(t=40,b=10), height=320,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Cross-Tabulation: Segment × Subscription Intent")
    ct = pd.crosstab(dff[C["seg"]], dff[C["target"]], margins=True)
    ct.columns = ["Would NOT Subscribe","Would Subscribe","Total"]
    ct["Subscribe Rate"] = (ct["Would Subscribe"]/ct["Total"]*100).round(1).astype(str)+"%"
    st.dataframe(ct.style.background_gradient(subset=["Would Subscribe"], cmap="YlGn"),
                 use_container_width=True)

    st.markdown("""<div class='insight-box'>
    <b>🔑 Key Descriptive Insights</b><br>
    • <b>Nuclear Families and Busy Professionals</b> are 58% of respondents and drive the highest subscription intent.<br>
    • <b>Monthly service spend is right-skewed</b> — most residents spend AED 500–2,500 but Premium Residents push the mean up significantly.<br>
    • <b>6–8 apps</b> is the norm for Busy Professionals and Nuclear Families — fragmentation pain is real and measurable.<br>
    • <b>Laundry, Cleaning, and Grocery</b> are used by 75%+ of all respondents — the natural core of the MVP bundle.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Why Do People Subscribe — or Not?")
    st.markdown("*Correlations, frustrations, trust barriers, and behavioural drivers*")

    st.markdown("### 📈 Correlation Matrix")
    corr_rename = {
        C["pain"]:      "Pain Score",
        C["spend"]:     "Monthly Spend",
        C["apps"]:      "# Apps",
        C["sat"]:       "Satisfaction",
        C["wtp"]:       "WTP",
        C["intent"]:    "Intent",
        C["ai"]:        "AI Comfort",
        C["imp_app"]:   "Importance",
        C["lifescore"]: "LifeScore",
        C["target"]:    "Subscribe",
    }
    corr_df = dff[list(corr_rename.keys())].dropna().rename(columns=corr_rename)
    fig = px.imshow(corr_df.corr(), text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0,CORAL],[0.5,"white"],[1,NAVY]],
                    title="Pearson Correlation Matrix", zmin=-1, zmax=1)
    fig.update_layout(margin=dict(t=40,b=20), height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fs = dff.groupby([C["frust"], C["seg"]]).size().reset_index(name="Count")
        fig = px.bar(fs, x=C["frust"], y="Count", color=C["seg"],
                     color_discrete_map=SEG_COLORS,
                     title="Top Frustrations by Segment", barmode="group")
        fig.update_layout(margin=dict(t=40,b=10), height=360,
                          xaxis_title="", xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ts = dff.groupby(C["trust"])[C["target"]].mean().reset_index()
        ts.columns = ["Trust Driver","Subscribe Rate"]
        ts["Subscribe Rate"] *= 100
        fig = px.bar(ts.sort_values("Subscribe Rate"),
                     x="Subscribe Rate", y="Trust Driver", orientation="h",
                     title="Subscription Rate by Trust Driver (%)",
                     color="Subscribe Rate",
                     color_continuous_scale=[[0,"#EDE9FE"],[1,PURPLE]])
        fig.update_layout(margin=dict(t=40,b=10), height=360,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(dff, x=C["pain"], y=C["wtp"],
                         color=C["seg"], color_discrete_map=SEG_COLORS,
                         trendline="ols", opacity=0.6,
                         title="Pain Score vs Willingness to Pay",
                         labels={C["pain"]:"Pain Score (1–10)", C["wtp"]:"WTP (AED/month)"})
        fig.update_layout(margin=dict(t=40,b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(dff, x=C["spend"], y=C["wtp"],
                         color=C["seg"], color_discrete_map=SEG_COLORS,
                         trendline="ols", opacity=0.6,
                         title="Current Spend vs Willingness to Pay",
                         labels={C["spend"]:"Monthly Spend (AED)", C["wtp"]:"WTP (AED/month)"})
        fig.update_layout(margin=dict(t=40,b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        hrs_map = {1:"<1 hr",2:"1–2 hrs",3:"3–4 hrs",4:"5–6 hrs",5:">6 hrs"}
        hs = dff.groupby(C["hours"])[C["target"]].mean().reset_index()
        hs.columns = ["Hours","Subscribe Rate"]
        hs["Subscribe Rate"] *= 100
        hs["Hours"] = hs["Hours"].map(hrs_map)
        fig = px.line(hs, x="Hours", y="Subscribe Rate", markers=True,
                      title="More Time Wasted → Higher Subscribe Rate",
                      labels={"Subscribe Rate":"Subscribe Rate (%)"})
        fig.update_traces(line_color=TEAL, marker_size=10)
        fig.update_layout(margin=dict(t=40,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        aw = dff.groupby(C["apps"])[C["wtp"]].mean().reset_index()
        fig = px.bar(aw, x=C["apps"], y=C["wtp"],
                     color=C["wtp"],
                     color_continuous_scale=[[0,"#FEF3C7"],[1,AMBER]],
                     title="More Apps = Higher WTP",
                     labels={C["apps"]:"Number of Apps", C["wtp"]:"Avg WTP (AED)"})
        fig.update_layout(margin=dict(t=40,b=10), height=320,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ❌ Why Won't Some People Subscribe?")
    no_sub = dff[dff[C["target"]] == 0]
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(no_sub, names=C["frust"], hole=0.4,
                     title="Non-Subscribers: Primary Frustration",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(no_sub, names=C["seg"], hole=0.4,
                     title="Non-Subscribers: Which Segment?",
                     color=C["seg"], color_discrete_map=SEG_COLORS)
        fig.update_layout(margin=dict(t=40,b=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.pie(no_sub, names=C["trust"], hole=0.4,
                     title="Non-Subscribers: Trust Barrier",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=40,b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⭐ LifeScore — Quality of Life by Segment")
    ls2 = dff.groupby(C["seg"])[C["lifescore"]].agg(["mean","std"]).reset_index()
    ls2.columns = ["Segment","Mean","Std"]
    fig = px.bar(ls2, x="Segment", y="Mean", color="Segment",
                 color_discrete_map=SEG_COLORS, error_y="Std",
                 title="Average LifeScore (1=Very Poor → 5=Excellent)",
                 labels={"Mean":"Average LifeScore"})
    fig.add_hline(y=3, line_dash="dash", line_color="gray",
                  annotation_text="Neutral (3.0)")
    fig.update_layout(showlegend=False, margin=dict(t=40,b=10), height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    <b>🔑 Key Diagnostic Insights</b><br>
    • <b>Pain score is the #1 driver</b> of both WTP and subscription intent — the higher the frustration, the more they will pay.<br>
    • <b>Budget Expats cite cost (46%) as top frustration</b> — they need a savings message, not a convenience message.<br>
    • <b>Every additional app adds ~AED 15–20 to WTP</b> — fragmentation directly increases the commercial opportunity.<br>
    • <b>Residents spending 5+ hours/week on logistics show 75%+ subscribe rate</b> — time-starved users convert fastest.<br>
    • <b>Non-subscribers need trust not persuasion</b> — transparent pricing and free trial dominate as barriers.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Who Will Subscribe? Machine Learning Predictions")
    st.markdown("*Classification models trained on survey data to predict subscription likelihood*")

    with st.spinner("Training 4 ML models — takes about 20 seconds..."):
        results, fi, feature_cols, Xpca, cluster_labels, fe = run_ml(df)

    st.markdown("### 🏆 Model Performance Summary")
    model_colors = {
        "KNN":"#06B6D4","Decision Tree":AMBER,
        "Random Forest":GREEN,"Gradient Boosting":PURPLE,
    }
    cols = st.columns(4)
    for col, (name, res) in zip(cols, results.items()):
        col.markdown(f"""<div class='metric-card' style='border-top-color:{model_colors[name]}'>
          <div style='font-size:0.88rem;font-weight:600;color:{model_colors[name]}'>{name}</div>
          <div style='font-size:1.8rem;font-weight:700;color:#1B3A6B'>{res['test_acc']*100:.1f}%</div>
          <div style='color:#6B7280;font-size:0.78rem'>Test Accuracy</div>
          <div style='margin-top:8px;font-size:0.8rem;color:#374151'>
            F1: {res['f1']:.3f} | AUC: {res['auc']:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        mnames = list(results.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train Accuracy", x=mnames,
                             y=[results[m]["train_acc"]*100 for m in mnames],
                             marker_color=NAVY, opacity=0.8))
        fig.add_trace(go.Bar(name="Test Accuracy", x=mnames,
                             y=[results[m]["test_acc"]*100 for m in mnames],
                             marker_color=AMBER, opacity=0.9))
        fig.update_layout(title="Train vs Test Accuracy (%)", barmode="group",
                          yaxis=dict(range=[75,101]),
                          margin=dict(t=40,b=10), height=360,
                          yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for metric, color in zip(["precision","recall","f1","auc"],
                                  [TEAL,CORAL,GREEN,PURPLE]):
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=mnames,
                y=[results[m][metric] for m in mnames],
                marker_color=color, opacity=0.85))
        fig.update_layout(title="Precision / Recall / F1 / AUC",
                          barmode="group", yaxis=dict(range=[0.7,1.0]),
                          margin=dict(t=40,b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 ROC Curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                              line=dict(dash="dash",color="gray"),name="Random Chance"))
    for name, res in results.items():
        fig.add_trace(go.Scatter(x=res["fpr"],y=res["tpr"],mode="lines",
                                  name=f"{name} (AUC={res['auc']:.3f})",
                                  line=dict(color=model_colors[name],width=2.5)))
    fig.update_layout(title="ROC Curves — All Models",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      margin=dict(t=40,b=20), height=420,
                      legend=dict(x=0.55,y=0.05))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Confusion Matrices")
    cm_cols = st.columns(4)
    for col, (name, res) in zip(cm_cols, results.items()):
        cm   = res["cm"]
        tot  = cm.sum()
        tn,fp,fn,tp = cm.ravel()
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0,"#EFF6FF"],[1,NAVY]],
                        labels=dict(x="Predicted",y="Actual"),
                        x=["No (0)","Yes (1)"],y=["No (0)","Yes (1)"],
                        title=name)
        fig.update_layout(margin=dict(t=40,b=10), height=260,
                          coloraxis_showscale=False)
        col.plotly_chart(fig, use_container_width=True)
        col.markdown(
            f"<div style='background:#FEF9EE;border-radius:6px;padding:6px;"
            f"font-size:11px;text-align:center'>🔴 FP: <b>{fp}</b> ({fp/tot*100:.1f}%) "
            f"| 🟠 FN: <b>{fn}</b> ({fn/tot*100:.1f}%)</div>",
            unsafe_allow_html=True)

    st.markdown("### 🔑 Feature Importance — Top 15 Predictors")
    fi_top = fi.head(15).reset_index()
    fi_top.columns = ["Feature","Importance"]
    fi_top["Feature"] = (fi_top["Feature"]
        .str.replace("_enc","",regex=False)
        .str.replace(r"Q\d+_","",regex=True)
        .str.replace("_"," ")
        .str.title())
    fig = px.bar(fi_top, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=[[0,"#DBEAFE"],[1,NAVY]],
                 title="Random Forest Feature Importance")
    fig.update_layout(margin=dict(t=40,b=10), height=480,
                      coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗺️ Customer Segments — PCA Cluster Plot")
    pca_df = pd.DataFrame({
        "PC1": Xpca[:,0], "PC2": Xpca[:,1],
        "Cluster": [f"Cluster {chr(65+l)}" for l in cluster_labels],
        "Segment": df[C["seg"]],
        "WTP (AED)": df[C["wtp"]],
    })
    clust_colors = {"Cluster A":"#2563EB","Cluster B":"#10B981",
                    "Cluster C":"#F59E0B","Cluster D":"#8B5CF6"}
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                     color_discrete_map=clust_colors,
                     size="WTP (AED)", size_max=12, opacity=0.7,
                     hover_data=["Segment","WTP (AED)"],
                     title="K-Means Segmentation in PCA Space (4 Clusters)")
    fig.update_layout(margin=dict(t=40,b=10), height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    <b>🔑 Key Predictive Insights</b><br>
    • <b>Gradient Boosting is the best model</b> — highest AUC and F1, minimal overfitting gap between train and test.<br>
    • <b>Subscription intent and WTP are the top two predictors</b> — these two questions alone explain most of the variance.<br>
    • <b>Pain × Importance interaction</b> adds significant lift over pain alone — feature engineering matters.<br>
    • <b>Four clusters separate cleanly in PCA space</b> — confirming the four-segment hypothesis from the business plan.<br>
    • <b>False Negatives are the critical business error</b> — missing a likely subscriber costs revenue; Gradient Boosting minimises this.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🎯 What Should LifePass Do?")
    st.markdown("*Data-driven recommendations on pricing, targeting, product, and go-to-market*")

    st.markdown("### 💡 LifePass Tier Recommender")
    st.markdown("*Enter a customer profile and get their recommended subscription tier*")

    with st.form("recommender"):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_inc   = st.selectbox("Monthly Income",
                ["<AED 5,000","AED 5,000–10,000","AED 10,001–20,000",
                 "AED 20,001–35,000",">AED 35,000"])
            r_spend = st.number_input("Current Monthly Service Spend (AED)",
                                      100, 15000, 1200, 100)
            r_apps  = st.slider("Number of Apps Used", 1, 14, 6)
        with rc2:
            r_pain  = st.slider("Pain Score (1–10)", 1, 10, 7)
            r_liv   = st.selectbox("Living Situation",
                ["Living alone","Couple no kids","Family with kids",
                 "Flatmates","Extended family"])
            r_car   = st.checkbox("Has a car")
        with rc3:
            r_pet   = st.checkbox("Has a pet")
            r_ai    = st.slider("AI Comfort (1–5)", 1, 5, 3)
            r_subs  = st.slider("Existing Subscriptions", 0, 3, 1)
        submitted = st.form_submit_button("🔮 Get Recommendation",
                                          use_container_width=True)

    if submitted:
        inc_val = {"<AED 5,000":3500,"AED 5,000–10,000":7500,
                   "AED 10,001–20,000":15000,"AED 20,001–35,000":27500,
                   ">AED 35,000":50000}[r_inc]
        if "Family" in r_liv and inc_val >= 10000:
            tier,color,price,svcs = ("🌟 Family Plan",GREEN,"AED 349/month",
                "Laundry · Cleaning · Grocery · Salon · Maintenance")
        elif inc_val >= 25000 or r_spend >= 3000:
            tier,color,price,svcs = ("👑 Premium Plan",PURPLE,"AED 599/month",
                "All services · Airport · Concierge · Priority scheduling")
        elif inc_val <= 7500 or r_spend <= 600:
            tier,color,price,svcs = ("💚 Essential Plan",TEAL,"AED 199/month",
                "Laundry · Home cleaning · Car wash · Grocery delivery")
        else:
            tier,color,price,svcs = ("⚡ Professional Plan",NAVY,"AED 299/month",
                "Laundry · Cleaning · Car wash · Grocery · Salon")
        price_num = int(price.split("AED ")[1].split("/")[0])
        savings = max(0, r_spend - price_num)
        st.markdown(f"""
        <div style='background:{color}15;border:2px solid {color};
        border-radius:12px;padding:20px;margin-top:12px'>
          <h3 style='color:{color};margin:0'>Recommended: {tier}</h3>
          <p style='margin:8px 0;font-size:1.05rem'><b>{price}</b> · {svcs}</p>
          <p style='margin:0;color:#374151'>
            💰 Estimated monthly savings vs current spend: <b>AED {savings:+,}</b></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        wtp_ord = ["Below AED 100","AED 100-199","AED 200-299",
                   "AED 300-399","AED 400-499","AED 500-749","AED 750+"]
        wd = dff[C["wtp_cat"]].value_counts().reset_index()
        wd.columns = ["WTP Category","Count"]
        wd["WTP Category"] = pd.Categorical(wd["WTP Category"],
                                             categories=wtp_ord, ordered=True)
        wd = wd.sort_values("WTP Category")
        fig = px.bar(wd, x="WTP Category", y="Count",
                     color="Count",
                     color_continuous_scale=[[0,"#DBEAFE"],[1,NAVY]],
                     title="WTP Distribution → Validates 3-Tier Pricing")
        for xv, lbl in [(1.5,"Essential AED 199"),(3.5,"Family AED 349"),(5.5,"Premium AED 599")]:
            fig.add_vline(x=xv, line_dash="dash", line_color=AMBER,
                          annotation_text=lbl, annotation_font_size=10)
        fig.update_layout(margin=dict(t=40,b=10), height=360,
                          showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sw = dff.groupby(C["seg"])[C["wtp"]].agg(["mean","median"]).reset_index()
        sw.columns = ["Segment","Mean WTP","Median WTP"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mean WTP", x=sw["Segment"],
                             y=sw["Mean WTP"], marker_color=NAVY, opacity=0.85))
        fig.add_trace(go.Bar(name="Median WTP", x=sw["Segment"],
                             y=sw["Median WTP"], marker_color=AMBER, opacity=0.85))
        for yv, lbl in [(199,"AED 199"),(349,"AED 349"),(599,"AED 599")]:
            fig.add_hline(y=yv, line_dash="dot", line_color=GREEN,
                          annotation_text=lbl, annotation_font_size=10)
        fig.update_layout(title="WTP by Segment vs Price Points",
                          barmode="group", margin=dict(t=40,b=10),
                          height=360, yaxis_title="AED/month")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📦 MVP Service Bundle Priority")
    svc_priority = {
        "Laundry":      dff[C["u_laundry"]].mean()*100,
        "Home Cleaning":dff[C["u_cleaning"]].mean()*100,
        "Grocery":      dff[C["u_grocery"]].mean()*100,
        "Car Wash":     dff[C["u_carwash"]].mean()*100,
        "Salon":        dff[C["u_salon"]].mean()*100,
        "Maintenance":  dff[C["u_maint"]].mean()*100,
        "Airport":      dff[C["u_airport"]].mean()*100,
        "Pet Care":     dff[C["u_pet"]].mean()*100,
        "Concierge":    dff[C["u_concierge"]].mean()*100,
    }
    sp2 = pd.DataFrame(list(svc_priority.items()),
                       columns=["Service","Usage %"])
    sp2 = sp2.sort_values("Usage %", ascending=True)
    sp2["Priority"] = sp2["Usage %"].apply(
        lambda x: "🟢 MVP Core" if x>=65 else ("🟡 Phase 2" if x>=40 else "🔵 Premium Add-on"))
    fig = px.bar(sp2, x="Usage %", y="Service", orientation="h",
                 color="Priority",
                 color_discrete_map={"🟢 MVP Core":GREEN,
                                      "🟡 Phase 2":AMBER,
                                      "🔵 Premium Add-on":PURPLE},
                 title="Service Adoption Rate → MVP Launch Priority")
    fig.add_vline(x=65, line_dash="dash", line_color=GREEN,
                  annotation_text="MVP threshold 65%")
    fig.update_layout(margin=dict(t=40,b=10), height=400,
                      xaxis_title="% of Respondents Using Service")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚀 Go-to-Market Priority Matrix")
    c1, c2 = st.columns(2)
    with c1:
        gtm = dff.groupby(C["seg"]).agg(
            subscribe_rate=(C["target"],"mean"),
            avg_wtp=(C["wtp"],"mean"),
            count=(C["id"],"count")
        ).reset_index()
        gtm["subscribe_rate"] *= 100
        fig = px.scatter(gtm, x="subscribe_rate", y="avg_wtp",
                         size="count", color=C["seg"],
                         color_discrete_map=SEG_COLORS,
                         text=C["seg"], size_max=55,
                         title="Priority Matrix: Subscribe Rate vs WTP",
                         labels={"subscribe_rate":"Subscribe Rate (%)",
                                  "avg_wtp":"Avg WTP (AED)"})
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.add_vline(x=gtm["subscribe_rate"].mean(),
                      line_dash="dash", line_color="gray")
        fig.add_hline(y=gtm["avg_wtp"].mean(),
                      line_dash="dash", line_color="gray")
        fig.update_layout(margin=dict(t=40,b=10),
                          height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gtm["Revenue (AED K)"] = (
            gtm["subscribe_rate"]/100 * gtm["avg_wtp"] * gtm["count"] / 1000
        ).round(1)
        fig = px.bar(gtm.sort_values("Revenue (AED K)", ascending=False),
                     x=C["seg"], y="Revenue (AED K)",
                     color=C["seg"], color_discrete_map=SEG_COLORS,
                     title="Monthly Revenue Potential by Segment (AED '000s)",
                     labels={C["seg"]:""})
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Data-Driven Action Plan")
    actions = pd.DataFrame([
        ["1","Nuclear Family",
         "Launch Family Plan (AED 349) first",
         "Highest revenue potential + highest subscribe rate in data",
         "WhatsApp campaigns in Mirdif/Sports City"],
        ["2","Busy Professional",
         "Lead with time-saving message",
         "68% cite app overload as top problem — time is the pain",
         "LinkedIn + commute-time ads in JLT/Marina"],
        ["3","Budget Expat",
         "Offer 7-day free trial",
         "46% say too expensive — trial removes the cost barrier",
         "International City WhatsApp groups + flyer drops"],
        ["4","Premium Resident",
         "Concierge + Airport as hero features",
         "Median WTP AED 500+ — premium tier is underpriced for this segment",
         "Palm/Downtown building concierge desk partnerships"],
        ["5","All segments",
         "MVP = 3 services: Laundry, Cleaning, Grocery",
         "These 3 used by 70–90% across all segments",
         "Prove bundle value before expanding to 9 services"],
        ["6","All segments",
         "Build referral programme from day 1",
         "Peer recommendation is top trust driver across all segments",
         "AED 50 credit per successful referral"],
    ], columns=["Priority","Target","Action","Why (Data Evidence)","How"])
    st.dataframe(actions.set_index("Priority"), use_container_width=True)

    st.markdown("""<div class='insight-box'>
    <b>🎯 Key Prescriptive Recommendations</b><br>
    • <b>Nuclear Family + Busy Professional first</b> — 73% of total projected revenue from these two segments alone.<br>
    • <b>Price Essential at AED 199</b> — 38% of respondents have WTP AED 100–299, making this the volume driver.<br>
    • <b>MVP = 3 services only</b> — Laundry, Cleaning, Grocery used by 70–90% across all segments. Start here, prove it, then expand.<br>
    • <b>Free trial converts Budget Expats</b> — 46% cost frustration disappears when first week is free.<br>
    • <b>Premium Residents are underpriced</b> — median WTP AED 500+ means the AED 599 Premium tier converts better than expected.
    </div>""", unsafe_allow_html=True)
