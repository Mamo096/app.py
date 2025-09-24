import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="CORD-19 Metadata Dashboard", layout="wide")

DATA_PATH = Path("data/sample_metadata.csv")

@st.cache_data
def load_data(path=DATA_PATH):
    """Load dataset and do light preprocessing.
    Expects a publish_time column. If missing, the column will be created as NaT.
    """
    df = pd.read_csv(path, low_memory=False)

    
    if 'publish_time' not in df.columns:
        df['publish_time'] = pd.NaT

    
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    
    df['year'] = df['publish_time'].dt.year
    df['month'] = df['publish_time'].dt.to_period('M')

    
    if 'journal' in df.columns:
        df['journal'] = df['journal'].fillna('(Unknown)').astype(str)
    else:
        df['journal'] = '(Unknown)'


    if 'title' in df.columns:
        df['title_short'] = df['title'].astype(str).str.slice(0, 150)
    else:
        df['title'] = ''
        df['title_short'] = ''

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Please place sample_metadata.csv in the data/ folder.")
    st.stop()

st.sidebar.header("Filters")
min_year = int(df['year'].dropna().min()) if df['year'].dropna().any() else 2020
max_year = int(df['year'].dropna().max()) if df['year'].dropna().any() else pd.Timestamp.now().year
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
selected_journal = st.sidebar.selectbox("Journal (Top 25 + Unknown)", options=['All'] + df['journal'].value_counts().head(25).index.tolist())
keyword = st.sidebar.text_input("Keyword in title/abstract")

mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
if selected_journal != 'All':
    mask &= (df['journal'] == selected_journal)
if keyword:
    keyword_low = keyword.lower()
    contains_title = df['title'].astype(str).str.lower().str.contains(keyword_low, na=False)
    
    if 'abstract' in df.columns:
        contains_abstract = df['abstract'].astype(str).str.lower().str.contains(keyword_low, na=False)
    else:
        contains_abstract = pd.Series(False, index=df.index)
    mask &= (contains_title | contains_abstract)

filtered = df[mask].copy()

st.title("📊 CORD-19 Metadata Dashboard")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Overview")
    st.markdown(f"*Total papers (selection):* {len(filtered):,}")
    st.markdown(f"*Date range:* {year_range[0]} — {year_range[1]}")

    
    st.subheader("Papers published over time")
    timeseries = filtered.dropna(subset=['publish_time']).groupby(filtered['publish_time'].dt.to_period('M')).size()
    if len(timeseries) == 0:
        st.info("No time-series data available for the current selection.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        timeseries.index = timeseries.index.to_timestamp()
        timeseries.plot(ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of papers')
        ax.set_title('Papers published per month')
        st.pyplot(fig)

    
    st.subheader("Top journals in selection")
    top_j = filtered['journal'].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_j.values, y=top_j.index, ax=ax2)
    ax2.set_xlabel('Number of papers')
    ax2.set_ylabel('Journal')
    st.pyplot(fig2)

with col2:
    st.header("Sample records")
    if len(filtered) == 0:
        st.write("No records to display with current filters.")
    else:
        st.dataframe(filtered[['title_short', 'publish_time', 'journal']].rename(columns={'title_short': 'Title (short)'}).head(50))

@st.cache_data
def to_csv_download(df_inner):
    return df_inner.to_csv(index=False).encode('utf-8')

csv = to_csv_download(filtered)
st.download_button(label='Download filtered data as CSV', data=csv, file_name='filtered_metadata.csv', mime='text/csv')

notebooks/EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_PATH = Path('../data/sample_metadata.csv')

df = pd.read_csv(DATA_PATH, low_memory=False)


print(df.shape)
df.head()


print(df.info())
print(df.isna().sum().sort_values(ascending=False).head(20))


if 'publish_time' in df.columns:
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
else:
    df['publish_time'] = pd.NaT


df['year'] = df['publish_time'].dt.year


papers_per_year = df['year'].value_counts().sort_index()
plt.figure(figsize=(10,4))
plt.plot(papers_per_year.index, papers_per_year.values)
plt.xlabel('Year')
plt.ylabel('Number of papers')
plt.title('Papers per Year')
plt.show()


if 'journal' in df.columns:
    top_journals = df['journal'].value_counts().head(20)
    plt.figure(figsize=(8,6))
    sns.barplot(y=top_journals.index, x=top_journals.values)
    plt.xlabel('Number of papers')
    plt.ylabel('Journal')
    plt.title('Top Journals')
    plt.show()


from collections import Counter
import re

texts = df['title'].dropna().astype(str).str.lower()
words = Counter()
for t in texts:
    tokens = re.findall(r"\b[a-z]{3,}\b", t)
    words.update(tokens)

for w, c in words.most_common(30):
    print(w, c)


clean_sample = df[['title','abstract','publish_time','journal']].copy()
clean_sample.to_csv('../data/sample_metadata_clean.csv', index=False)