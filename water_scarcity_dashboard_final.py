import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import base64

# Set page configuration
st.set_page_config(page_title="Water Scarcity Dashboard - Kenya", layout="wide")

# Add Banner with Fallback (Full Width, Adjusted Height)
banner_path = "banner.jpg"
if os.path.exists(banner_path) and os.path.getsize(banner_path) > 0:
    with open(banner_path, "rb") as img_file:
        encoded_banner = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .banner-container img {{
            width: 100% !important;
            height: 60vh !important;  /* Updated height to 60% of viewport */
            object-fit: cover;       /* crop without distortion */
        }}
        </style>
        <div class="banner-container">
            <img src="data:image/png;base64,{encoded_banner}" alt="Water Scarcity Banner">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Banner image 'banner.jpg' not found. Using fallback URL. Please place banner.jpg in the directory.")
    st.image("https://images.unsplash.com/photo-1715932809-4708d6d7b2a8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1920&q=80", 
             use_column_width=True, caption="Fallback Water Scarcity Image (Unsplash)")

# Load synthetic dataset
try:
    df = pd.read_csv("synthetic_water_scarcity_data.csv")
except FileNotFoundError:
    st.error("Error: synthetic_water_scarcity_data.csv not found. Please create or place the file in C:\\Users\\user\\OneDrive\\Desktop\\Hackathon 2025\\WaterScarcity. A sample format is:\nRegion,Date,Water_Stress_Level (%),Rainfall (mm),Crop_Recommendation,Irrigation_Schedule\nNairobi,2024-01-01,45.5,600,Maize,Weekly")
    st.stop()

# Aggregate to annual data for dashboard display
annual_df = df.groupby("Region").agg({
    "Water_Stress_Level (%)": "mean",
    "Rainfall (mm)": "sum",
    "Crop_Recommendation": "first",
    "Irrigation_Schedule": "first"
}).reset_index()
annual_df = annual_df.rename(columns={"Rainfall (mm)": "Rainfall_2025 (mm)"})

# Approximate coordinates for Kenyan counties
coords = {
    "Nairobi": [-1.286389, 36.817223],
    "Makueni": [-1.8037, 37.6203],
    "Kitui": [-1.3741, 38.0106],
    "Machakos": [-1.5181, 37.2662],
    "Embu": [-0.5388, 37.4506]
}
default_coord = [-0.0236, 37.9062]

def get_latitude(region):
    return coords.get(region, default_coord)[0]

def get_longitude(region):
    return coords.get(region, default_coord)[1]

annual_df["Latitude"] = annual_df["Region"].map(get_latitude)
annual_df["Longitude"] = annual_df["Region"].map(get_longitude)

# Filter out rows with NaN in Water_Stress_Level (%)
annual_df = annual_df.dropna(subset=["Water_Stress_Level (%)"])

# Train a decision tree model for irrigation prediction
X = annual_df[["Rainfall_2025 (mm)", "Water_Stress_Level (%)"]]
y = annual_df["Irrigation_Schedule"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Function to predict irrigation using the ML model
def predict_irrigation(rainfall, water_stress):
    prediction = clf.predict([[rainfall, water_stress]])
    return le.inverse_transform(prediction)[0]

# Impact data for visualization
impact_data = pd.DataFrame({
    "Metric": ["Current Yield", "Projected Yield (+20%)", "Projected Yield (+30%)"],
    "Value (tonnes)": [100000, 120000, 130000],
    "People Fed (thousands)": [571, 686, 743]  # Average: 100,000 / 0.175 = 571,429 (~571k), 120,000 / 0.175 = 685,714 (~686k), 130,000 / 0.175 = 742,857 (~743k)
})

# Use local image files with validation
sdg2_icon = "sdg2.png"
sdg6_icon = "sdg6.png"
sdg13_icon = "sdg13.png"

if not all(os.path.exists(f) and os.path.getsize(f) > 0 for f in [sdg2_icon, sdg6_icon, sdg13_icon]):
    st.warning("One or more SDG icon files are missing or empty. Please replace sdg2.png, sdg6.png, and sdg13.png with valid PNG files.")

# Landing Page
st.title("Water Scarcity Dashboard")
st.markdown("**Empowering Kenyan farmers with AI to combat water scarcity and secure food systems.**")

# The Problem
st.header("The Problem")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(sdg13_icon, width=120, caption="SDG 13: Climate Action")  # Increased from 100 to 120 (20% more)
with col2:
    st.image(sdg2_icon, width=120, caption="SDG 2: Zero Hunger")      # Increased from 100 to 120 (20% more)
with col3:
    st.image(sdg6_icon, width=120, caption="SDG 6: Clean Water")      # Increased from 100 to 120 (20% more)
st.markdown("""
Kenya loses **125 billion KES annually** due to drought-related crop losses (World Bank).  
**80% of farmland** is rainfed, with semi-arid regions like Makueni and Kitui facing water stress levels up to **80–90%** in dry seasons. This threatens **SDG 13 (Climate Action)**, **SDG 2 (Zero Hunger)**, and **SDG 6 (Clean Water)**.  
*Data: Synthetic dataset based on World Bank, FAO AQUASTAT, KNBS, CHIRPS, Kenya Open Data.*
""")

# Our Solution
st.header("Our Solution")
st.markdown("""
Our AI-powered dashboard predicts water stress zones, suggests optimal irrigation schedules, and recommends drought-resilient crops → directly advancing SDG 13, 2, 6.

**AI Magic**:
- **Time-series climate prediction**: Forecasts rainfall and drought patterns.
- **Recommender system**: Suggests irrigation strategies and crops based on local conditions.
""")

# Demo Section
st.header("Demo Section")
st.markdown("Interactive map/chart showcasing AI-driven insights with dummy data.")

tab1, tab2 = st.tabs(["Select Region", "Prediction Explorer"])

with tab1:
    region = st.selectbox("Select Your Region", annual_df["Region"])
    selected_data = annual_df[annual_df["Region"] == region]
    rainfall = selected_data["Rainfall_2025 (mm)"].iloc[0]
    water_stress = selected_data["Water_Stress_Level (%)"].iloc[0]
    crop = selected_data["Crop_Recommendation"].iloc[0]
    irrigation = predict_irrigation(rainfall, water_stress)
    st.subheader(f"Recommendations for {region}")
    st.write(f"**Average Water Stress Level (2024–2025)**: {water_stress:.1f}%")
    st.write(f"**Total Rainfall (2024–2025)**: {rainfall:.0f} mm")
    st.write(f"**Recommended Crop**: {crop}")
    st.write(f"**AI-Predicted Irrigation Schedule**: {irrigation}")

with tab2:
    st.subheader("Prediction Explorer")
    custom_rainfall = st.slider("Annual Rainfall (mm)", 100, 1200, 500)
    custom_stress = st.slider("Water Stress Level (%)", 0, 100, 50)
    custom_irrigation = predict_irrigation(custom_rainfall, custom_stress)
    custom_crop = "Sorghum" if custom_rainfall < 500 else "Maize" if custom_rainfall < 800 else "Beans"
    st.write(f"**Custom Input Results**")
    st.write(f"**Water Stress Level**: {custom_stress}%")
    st.write(f"**Annual Rainfall**: {custom_rainfall} mm")
    st.write(f"**Recommended Crop**: {custom_crop}")
    st.write(f"**AI-Predicted Irrigation Schedule**: {custom_irrigation}")

# Visualizations
st.subheader("Interactive Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(annual_df, x="Region", y="Water_Stress_Level (%)", 
                 color="Water_Stress_Level (%)", 
                 title="Average Water Stress (2024–2025)",
                 labels={"Water_Stress_Level (%)": "Water Stress (%)"},
                 color_continuous_scale="Reds",
                 text="Water_Stress_Level (%)")
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    region_monthly = df[df["Region"] == region]
    fig_trend = px.line(region_monthly, x="Date", y="Water_Stress_Level (%)", 
                        title=f"Monthly Water Stress in {region} (2024–2025)",
                        labels={"Water_Stress_Level (%)": "Water Stress (%)"})
    st.plotly_chart(fig_trend, use_container_width=True)

# New Heatmap Visual
st.subheader("Water Stress Heatmap")
fig_heat = px.density_heatmap(df, x="Date", y="Region", z="Water_Stress_Level (%)", title="Water Stress Over Time")
st.plotly_chart(fig_heat, use_container_width=True)

# Interactive Map (Plotly scatter_mapbox with Debug)
st.subheader("Interactive Map")
st.write("Debug: Checking map data...")
st.write(f"Data sample: {annual_df[['Region', 'Latitude', 'Longitude', 'Water_Stress_Level (%)']].to_string()}")
try:
    fig_map = px.scatter_mapbox(
        annual_df, lat="Latitude", lon="Longitude",
        hover_name="Region", hover_data=["Water_Stress_Level (%)"],
        color="Water_Stress_Level (%)", size="Water_Stress_Level (%)",
        color_continuous_scale="Reds", zoom=6,
        height=700, mapbox_style="open-street-map", title="Water Stress Map"
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering map: {str(e)}. Check Latitude/Longitude data, internet connection, or plotly installation.")
    if os.path.exists("map_fallback.png") and os.path.getsize("map_fallback.png") > 0:
        st.image("map_fallback.png", caption="Fallback Map (Interactive Map Failed)", width=700)
    else:
        st.warning("Fallback image 'map_fallback.png' not found. Please create it.")

# Impact
st.header("Impact")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(sdg13_icon, width=120, caption="SDG 13: Climate Action")  # Increased from 100 to 120 (20% more)
with col2:
    st.image(sdg2_icon, width=120, caption="SDG 2: Zero Hunger")      # Increased from 100 to 120 (20% more)
with col3:
    st.image(sdg6_icon, width=120, caption="SDG 6: Clean Water")      # Increased from 100 to 120 (20% more)

# Interactive Impact Metrics
st.subheader("Impact Metrics Explorer")
impact_data = pd.DataFrame({
    "Metric": ["Current Yield", "Projected Yield (+20%)", "Projected Yield (+30%)"],
    "Value (tonnes)": [100000, 120000, 130000],
    "People Fed (thousands)": [571, 686, 743]  # Average: 100,000 / 0.175 = 571,429 (~571k), 120,000 / 0.175 = 685,714 (~686k), 130,000 / 0.175 = 742,857 (~743k)
})

# Interactive slider to explore impact with range
selected_yield_increase = st.slider("Select Yield Increase (%)", 0, 30, 20, step=5)
consumption_rate = st.slider("Consumption Rate (kg/person/year)", 150, 200, 175, step=25) / 1000  # Convert to tonnes
projected_yield = 100000 * (1 + selected_yield_increase / 100)
people_fed = (projected_yield / consumption_rate) / 1000  # Convert to thousands
st.write(f"**Projected Yield**: {projected_yield:,.0f} tonnes")
st.write(f"**People Fed**: {people_fed:.0f} thousand (based on {int(consumption_rate * 1000)} kg/person/year)")

# Visual representation with custom blue shades
fig_impact = px.bar(impact_data, x="Metric", y=["Value (tonnes)", "People Fed (thousands)"], 
                    title="Impact of Adoption", barmode="stack", height=400,
                    labels={"value": "Amount", "variable": "Category"})
fig_impact.update_traces(marker_color=['#ADD8E6', '#87CEEB'],  # Blue shades: Light blue for tonnes, Sky blue for people
                        hovertemplate="%{y:,.0f} %{x}")
fig_impact.update_layout(
    legend_title_text="Categories",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)
st.plotly_chart(fig_impact, use_container_width=True)

st.markdown("""
**Key Benefits**:
- **Yield Increase**: Up to 30% more crops with AI adoption.
- **Food Security**: Feeds thousands more with optimized water use.
- **Economic Savings**: Saves billions in drought losses.
- **SDG Alignment**: Supports SDG 13, 2, and 6.
""")

# Add Report Download Button
if os.path.exists("water_scarcity_report.pdf") and os.path.getsize("water_scarcity_report.pdf") > 0:
    with open("water_scarcity_report.pdf", "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="water_scarcity_report.pdf">Download Impact Report</a>'
        st.markdown(href, unsafe_allow_html=True)
else:
    st.warning("Impact Report PDF not found. Please compile water_scarcity_report.tex to generate it.")

# Future Potential
st.header("Future Potential")
st.markdown("""
Scalable across Africa, where **65% of agriculture** is rainfed. Partnerships with NGOs and governments can deploy the tool to millions of farmers, advancing climate resilience and food security.
""")

# Call to Action
st.header("Call to Action")
st.markdown("""
**With partners, we can scale this to farmers** across Kenya and Africa. Let's collaborate to build a resilient future!
""")