from preswald import text, plotly, connect, get_df, table
import pandas as pd
import plotly.express as px
from preswald import selectbox

text("# Student depression and mental health analysis")

# Load the CSV
connect() # load in all sources, which by default is the sample_csv
df = get_df('sample_csv')

# data cleaning
df = df.dropna()
df = df[df["Sleep Duration"] != "Others"]
df = df[df["Dietary Habits"] != "Others"]
df = df[df["Financial Stress"] != "?"]
# only looking at students so remove those that are not students
df = df[df['Profession'].isin(['Student'])]
# drop other columns not related to students
df.drop(columns=['Profession', 'Work Pressure', 'Job Satisfaction'], inplace=True)


# numerical columns
numeric_cols = ['Age', 'Academic Pressure', 'CGPA','Study Satisfaction',
                'Work/Study Hours', 'Have you ever had suicidal thoughts ?',
                'Family History of Mental Illness', 'Financial Stress', 'Depression']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

sleep_mapping = {
    "'Less than 5 hours'": 4,
    "'5-6 hours'": 5.5,
    "'7-8 hours'": 7.5,
    "'More than 8 hours'": 9,
}
df['SleepHours'] = df['Sleep Duration'].map(sleep_mapping)

corr_cols = numeric_cols + ['SleepHours']
correlations = df[corr_cols].corr()['Depression'].drop('Depression')


text("## Lets start off by checking how correlated the features are with depression")

# sorted bar plot
fig_corr = px.bar(
    correlations.sort_values(ascending=False),
    orientation='h',
    labels={'value': 'Correlation with Depression', 'index': 'Feature'},
    title='Correlation of Features with Depression'
)

fig_corr.update_layout(yaxis={'categoryorder':'total ascending'})
plotly(fig_corr)

text("## Whats weird is that CGPA and family history of mental illness are not correlated with depression...I wonder if we can find any patterns in the data that show otherwise")

# break up CGPA into bins
df["CGPA Bin"] = pd.cut(df["CGPA"], bins=[0, 6, 7, 8, 9, 10], labels=["<6", "6-7", "7-8", "8-9", "9-10"])

df_grouped = df.groupby(["Family History of Mental Illness", "CGPA Bin"]).agg({
    "Depression": "mean"
}).reset_index()

fig_heat = px.density_heatmap(
    df_grouped,
    x="CGPA Bin",
    y="Family History of Mental Illness",
    z="Depression",
    color_continuous_scale="Reds",
    title="Depression Rate by CGPA Bin and Family History"
)

plotly(fig_heat)

text("## aha! So it seems like family history and CGPA are correlated with depression when we look at them together")

text("## Now lets finish off with an interactive plot")

available_features = [
    "SleepHours",
    "CGPA",
    "Academic Pressure",
    "Study Satisfaction", 
    "Financial Stress",
    "Work/Study Hours"
]

# let user select x and y axis features
x_col = selectbox("Select X-axis feature", available_features, default="CGPA")
y_col = selectbox("Select Y-axis feature", available_features, default="Academic Pressure")

# group by the selected features and calculate the mean depression rate and count
grouped = (
    df.groupby([x_col, y_col])
    .agg(mean_depression=("Depression", "mean"), count=("Depression", "size"))
    .reset_index()
)

# plot the mean depression rate vs the selected features
fig = px.scatter(
    grouped,
    x=x_col,
    y=y_col,
    color="mean_depression",
    size="count",
    color_continuous_scale="Viridis",
    labels={"mean_depression": "Avg Depression"},
    title=f"{y_col} vs {x_col} — Colored by Avg Depression Rate",
    height=600
)

fig.update_traces(marker=dict(sizemode='area', sizeref=2.*max(grouped["count"])/(40.**2), sizemin=4))

plotly(fig)

