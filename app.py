import streamlit as st
import pandas as pd
import time
from datetime import datetime
t=time.time()
d=datetime.fromtimestamp(t).strftime("%d - %m - %Y")
ts=datetime.fromtimestamp(t).strftime("%H: - %M - %S")
df=pd.read_csv("attendance/attandance_"+d+".csv")
st.dataframe(df.style.highlight_max(axis=0))