import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Clinic No-Show Dashboard",
    layout="wide"
)

# -----------------------------
# COLOR PALETTE
# -----------------------------
DARK_GREEN = "#2F4F3E"
MEDIUM_GREEN = "#4F7A5A"
SAGE_GREEN = "#8FB996"
LIGHT_GREEN = "#DDEBDD"
OFF_WHITE = "#F7FAF7"

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown(f"""
<style>
    .stApp {{
        background-color: {OFF_WHITE};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {LIGHT_GREEN};
    }}

    h1, h2, h3 {{
        color: {DARK_GREEN};
    }}

    p, div, label {{
        color: {DARK_GREEN};
    }}

    [data-testid="stMetricValue"] {{
        color: {DARK_GREEN};
        font-size: 2.2rem;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MEDIUM_GREEN};
        font-size: 1rem;
    }}

    .stAlert {{
        background-color: {LIGHT_GREEN};
        border: 1px solid {SAGE_GREEN};
        color: {DARK_GREEN};
        border-radius: 12px;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    div[data-baseweb="select"] > div {{
        background-color: white !important;
        border-color: {SAGE_GREEN} !important;
        color: {DARK_GREEN} !important;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("outputs/cleaned_data.csv")

df = load_data()

# -----------------------------
# CLEAN / STANDARDIZE DISPLAY FIELDS
# -----------------------------
if "AgeGroup" in df.columns:
    df["AgeGroup"] = df["AgeGroup"].astype(str)

if "WaitCategory" in df.columns:
    df["WaitCategory"] = df["WaitCategory"].astype(str)

# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("What Predicts Missed Medical Appointments?")
st.subheader("A Data-Driven Analysis of Clinic No-Show Risk and Operational Interventions")

st.write(
    "This dashboard explores which patient and scheduling factors are associated "
    "with missed medical appointments. The goal is to identify patterns clinics "
    "can use to improve efficiency, reduce unused appointment slots, and support "
    "more stable revenue planning."
)

st.subheader("Data and Preparation")
st.write(
    "This project uses a public medical appointment dataset containing patient, "
    "scheduling, and appointment attendance variables. Data preprocessing included "
    "datetime conversion, wait-time calculation, age filtering, duplicate removal, "
    "and feature engineering for appointment weekday, age group, and wait-time category."
)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

# Gender filter
if "Gender" in filtered_df.columns:
    gender_options = sorted(filtered_df["Gender"].dropna().unique().tolist())
    selected_gender = st.sidebar.multiselect(
        "Gender",
        gender_options,
        default=gender_options
    )
    if selected_gender:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_gender)]

# Weekday filter
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
if "AppointmentWeekday" in filtered_df.columns:
    weekday_options = [day for day in weekday_order if day in filtered_df["AppointmentWeekday"].dropna().unique()]
    selected_days = st.sidebar.multiselect(
        "Appointment Weekday",
        weekday_options,
        default=weekday_options
    )
    if selected_days:
        filtered_df = filtered_df[filtered_df["AppointmentWeekday"].isin(selected_days)]

# Age group filter
age_order = ["0-18", "19-35", "36-50", "51-65", "66+"]
if "AgeGroup" in filtered_df.columns:
    age_options = [age for age in age_order if age in filtered_df["AgeGroup"].dropna().unique()]
    selected_ages = st.sidebar.multiselect(
        "Age Group",
        age_options,
        default=age_options
    )
    if selected_ages:
        filtered_df = filtered_df[filtered_df["AgeGroup"].isin(selected_ages)]

# Wait category filter
wait_order = ["0-2 days", "3-7 days", "8-14 days", "15-30 days", "31+ days"]
if "WaitCategory" in filtered_df.columns:
    wait_options = [w for w in wait_order if w in filtered_df["WaitCategory"].dropna().unique()]
    selected_waits = st.sidebar.multiselect(
        "Wait Time Category",
        wait_options,
        default=wait_options
    )
    if selected_waits:
        filtered_df = filtered_df[filtered_df["WaitCategory"].isin(selected_waits)]

# SMS filter
if "SMS_received" in filtered_df.columns:
    sms_options = sorted(filtered_df["SMS_received"].dropna().unique().tolist())
    selected_sms = st.sidebar.multiselect(
        "SMS Received",
        sms_options,
        default=sms_options
    )
    if selected_sms:
        filtered_df = filtered_df[filtered_df["SMS_received"].isin(selected_sms)]

# -----------------------------
# KPI METRICS
# -----------------------------
st.info(
    "Key insight: About 28.5% of appointments in this dataset were missed. "
    "This indicates that no-shows are a meaningful clinic operations problem "
    "with implications for efficiency, staffing, and revenue stability."
)

col1, col2 = st.columns(2)
col1.metric("Appointments", f"{len(filtered_df):,}")
col2.metric("No-Show Rate", f"{filtered_df['NoShowFlag'].mean():.1%}")

# -----------------------------
# CHART SECTION HEADER
# -----------------------------
st.subheader("Attendance Patterns and Operational Risk Factors")

# -----------------------------
# CHART 1 + 2
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if "AppointmentWeekday" in filtered_df.columns:
        weekday_rates = (
            filtered_df.groupby("AppointmentWeekday", as_index=False)["NoShowFlag"]
            .mean()
        )

        weekday_rates["AppointmentWeekday"] = pd.Categorical(
            weekday_rates["AppointmentWeekday"],
            categories=weekday_order,
            ordered=True
        )
        weekday_rates = weekday_rates.sort_values("AppointmentWeekday")

        fig1 = px.bar(
            weekday_rates,
            x="AppointmentWeekday",
            y="NoShowFlag",
            title="No-Show Rate by Appointment Weekday",
            labels={
                "AppointmentWeekday": "Weekday",
                "NoShowFlag": "No-Show Rate"
            },
            color_discrete_sequence=[DARK_GREEN]
        )
        fig1.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=15, color=DARK_GREEN),
            title_font=dict(size=20, color=DARK_GREEN)
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if "AgeGroup" in filtered_df.columns:
        age_rates = (
            filtered_df.groupby("AgeGroup", as_index=False)["NoShowFlag"]
            .mean()
        )

        age_rates["AgeGroup"] = pd.Categorical(
            age_rates["AgeGroup"],
            categories=age_order,
            ordered=True
        )
        age_rates = age_rates.sort_values("AgeGroup")

        fig2 = px.bar(
            age_rates,
            x="AgeGroup",
            y="NoShowFlag",
            title="No-Show Rate by Age Group",
            labels={
                "AgeGroup": "Age Group",
                "NoShowFlag": "No-Show Rate"
            },
            color_discrete_sequence=[MEDIUM_GREEN]
        )
        fig2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=15, color=DARK_GREEN),
            title_font=dict(size=20, color=DARK_GREEN)
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# CHART 3 + 4
# -----------------------------
col3, col4 = st.columns(2)

with col3:
    if "WaitDays" in filtered_df.columns:
        plot_df = filtered_df.copy()
        plot_df["OutcomeLabel"] = plot_df["NoShowFlag"].map({
            0: "Attended",
            1: "No-Show"
        })

        fig3 = px.box(
            plot_df,
            x="OutcomeLabel",
            y="WaitDays",
            title="Wait Days by Appointment Outcome",
            labels={
                "OutcomeLabel": "Appointment Outcome",
                "WaitDays": "Wait Days"
            },
            color="OutcomeLabel",
            color_discrete_map={
                "Attended": MEDIUM_GREEN,
                "No-Show": SAGE_GREEN
            }
        )
        fig3.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=15, color=DARK_GREEN),
            title_font=dict(size=20, color=DARK_GREEN),
            legend_title_text=""
        )
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "SMS_received" in filtered_df.columns:
        sms_rates = (
            filtered_df.groupby("SMS_received", as_index=False)["NoShowFlag"]
            .mean()
        )

        fig4 = px.bar(
            sms_rates,
            x="SMS_received",
            y="NoShowFlag",
            title="No-Show Rate by SMS Reminder Status",
            labels={
                "SMS_received": "SMS Received (0 = No, 1 = Yes)",
                "NoShowFlag": "No-Show Rate"
            },
            color_discrete_sequence=[LIGHT_GREEN]
        )
        fig4.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=15, color=DARK_GREEN),
            title_font=dict(size=20, color=DARK_GREEN)
        )
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# CHART 5
# -----------------------------
if "WaitCategory" in filtered_df.columns:
    wait_rates = (
        filtered_df.groupby("WaitCategory", as_index=False)["NoShowFlag"]
        .mean()
    )

    wait_rates["WaitCategory"] = pd.Categorical(
        wait_rates["WaitCategory"],
        categories=wait_order,
        ordered=True
    )
    wait_rates = wait_rates.sort_values("WaitCategory")

    fig5 = px.bar(
        wait_rates,
        x="WaitCategory",
        y="NoShowFlag",
        title="No-Show Rate by Scheduling Lead Time",
        labels={
            "WaitCategory": "Wait Time Category",
            "NoShowFlag": "No-Show Rate"
        },
        color_discrete_sequence=[SAGE_GREEN]
    )
    fig5.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=15, color=DARK_GREEN),
        title_font=dict(size=20, color=DARK_GREEN)
    )
    st.plotly_chart(fig5, use_container_width=True)

st.subheader("Results Summary")

st.write(
    "The analysis found that missed appointments were common, with about 28.5% of visits "
    "in the dataset recorded as no-shows. No-show rates varied across scheduling and patient "
    "subgroups, suggesting that missed appointments are not distributed evenly across the clinic population."
)

st.write(
    "Operational factors such as appointment weekday, scheduling lead time, and reminder status "
    "appeared to be associated with attendance patterns. In particular, the wait-time visualizations "
    "suggest that scheduling delay may play an important role in no-show risk."
)

st.write(
    "A class-balanced logistic regression model improved detection of missed appointments compared with "
    "a baseline model, identifying about 57% of actual no-shows. Although the model produced many false "
    "positives, it still demonstrated that predictive analytics could help clinics identify higher-risk "
    "appointments for targeted intervention."
)

st.write(
    "Overall, the results support the business motivation of the project: no-shows are a meaningful "
    "operational problem, and data-driven targeting may help clinics improve efficiency and reduce lost capacity."
)
st.subheader("Model Performance")

st.write(
    "A class-balanced logistic regression model was used to improve detection of "
    "missed appointments. The balanced model achieved about 55.1% overall accuracy "
    "and identified about 57% of actual no-shows. Although this lowered total accuracy "
    "compared with a baseline model, it was more useful for the real operational goal: "
    "flagging appointments at risk of being missed."
)

st.write(
    "This result suggests that no-show prediction should be used as a screening tool "
    "for targeted intervention rather than as a fully automated scheduling decision system."
)

# -----------------------------
# BUSINESS RECOMMENDATIONS
# -----------------------------
st.subheader("Recommendations for Clinics")

st.markdown(
    """
1. **Target higher-risk appointments with extra outreach** such as reminder calls or follow-up texts.  
2. **Reduce long scheduling lead times when possible**, since longer waits may contribute to missed appointments.  
3. **Use prediction to support staff decisions**, not replace them, because false positives remain substantial.  
4. **Monitor subgroup patterns** by age, weekday, and scheduling delay to improve resource allocation over time.
"""
)

# -----------------------------
# PROJECT TAKEAWAY
# -----------------------------
st.subheader("Project Takeaway")

st.write(
    "Missed medical appointments are not just a patient behavior issue; they are also "
    "an operational and financial challenge for clinics. By identifying patterns in "
    "appointment attendance, healthcare organizations can design more targeted, data-driven "
    "strategies to improve efficiency and reduce unused clinical capacity."
)
