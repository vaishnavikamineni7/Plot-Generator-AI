import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama.llms import OllamaLLM

st.title("Plot Generator AI.")
# Initialize the LLM once
llm = OllamaLLM(model="gemma3")

# File uploader
data = st.file_uploader("Upload data", type="csv")

# Submit button to process data
if st.button("Submit"):
    if data is not None:
        df = pd.read_csv(data)
        st.session_state.df = df
        st.session_state.agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
        )
        st.session_state.data_uploaded = True
    else:
        st.warning("Please upload a CSV file.")

# If data was uploaded, show preview and accept query
if st.session_state.get("data_uploaded", False):
    st.subheader("📊 Table Preview:")
    st.dataframe(st.session_state.df)

    user_query = st.text_input("Ask something about your data (e.g., 'Plot sales over time')")
    query_sub=st.button("Submit Query")
    if query_sub:
        with st.spinner("Running analysis..."):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                response = st.session_state.agent.run(user_query)

            if response:
                st.write(response)

            st.pyplot(plt.gcf())
            plt.clf()
