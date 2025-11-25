import numpy as np
import pandas as pd
import streamlit as st

from code_generator import generate_python_code
from load_data import load_data
from safety_check import is_code_safe

# =============================
# Load financial DataFrame
# =============================
df = load_data()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Financial Q&A (Code-Gen)", page_icon="💹")

st.title("💹 Financial Data Q&A")
st.write(
    "Ask a question about the dataset. The app will use ChatGPT to generate "
    "Python code, run it against the DataFrame, and show the answer."
)

with st.expander("Preview of data"):
    st.dataframe(df.head())

st.markdown("**Columns:** " + ", ".join(df.columns))

question = st.text_area(
    "Ask a question:",
    placeholder='E.g. "What is the average price of AAPL?" or "Show me MSFT prices after 2025-01-10".',
    height=100,
)

if st.button("Run query"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Generating code and running query..."):
            # 1. Generate code with ChatGPT
            code = generate_python_code(question, df)

            st.subheader("Generated Python code")
            st.code(code, language="python")

            # 2. Safety check
            if not is_code_safe(code):
                st.error("The generated code contains disallowed operations and was not executed.")
            else:
                # 3. Execute code in a restricted namespace
                local_vars = {"df": df, "pd": pd, "np": np}
                try:
                    exec(code, {}, local_vars)
                except Exception as e:
                    st.error(f"Error while executing generated code: {e}")
                else:
                    # 4. Display result
                    answer = local_vars.get("answer", None)
                    result_df = local_vars.get("result_df", None)

                    st.subheader("Answer")
                    if answer is not None:
                        st.success(str(answer))
                    elif result_df is not None:
                        st.dataframe(result_df)
                    else:
                        st.info(
                            "The code ran, but it did not set an `answer` or `result_df` "
                            "variable. You may want to tighten the prompt."
                        )
