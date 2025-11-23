import os
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from load_data import load_data

# =============================
# 0. OpenAI client
# =============================
# Make sure OPENAI_API_KEY is set in your environment or Streamlit secrets.
client = OpenAI()  # reads OPENAI_API_KEY by default


# =============================
# 1. Sample financial DataFrame
# =============================

df = load_data()


# =============================
# 2. Helper: clean fenced code
# =============================
def clean_code_block(text: str) -> str:
    """
    Remove ```python ... ``` fences if the model returns them.
    """
    text = text.strip()

    if text.startswith("```"):
        # strip leading/trailing ```
        text = text.strip("`").strip()
        # drop 'python' or similar on first line if present
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("python"):
            lines = lines[1:]
        text = "\n".join(lines).strip()

    return text


# =============================
# 3. Code generator using ChatGPT
# =============================
def generate_python_code(question: str, df: pd.DataFrame) -> str:
    """
    Use ChatGPT to generate Python code that queries `df`.

    The code must:
      - Only use df, pandas as pd, and numpy as np (already in scope).
      - NOT import modules or access files/network/environment.
      - End by defining either:
          - `answer` (scalar or short string), or
          - `result_df` (a pandas DataFrame).
    """

    # Build a simple schema description
    schema_lines = [f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    schema_str = "\n".join(schema_lines)

    system_prompt = dedent(
        """
        You are an expert Python data analyst.
        Your task is to write SAFE Python code that analyzes a pandas DataFrame called `df`.

        Requirements:
        - The DataFrame `df` is already defined.
        - You may assume:
            import pandas as pd
            import numpy as np
          are already done.
        - DO NOT:
            * import anything
            * read or write files
            * access the network
            * use environment variables
            * use exec, eval, or subprocess
        - Use only `df`, `pd`, and `np`.

        Output format:
        - Write ONLY Python code, no explanations or comments.
        - Do NOT wrap the code in backticks.
        - At the end of your code, you must define either:
            * `answer`  -> for a scalar or short text answer
            * OR `result_df` -> a pandas DataFrame to display.

        If the question is ambiguous, make a reasonable assumption and proceed.
        """
    ).strip()

    # A couple of few-shot examples to steer the style
    examples = dedent(
        """
        DataFrame schema:
        Date: datetime64[ns]
        Stock: object
        Price: float64

        Example 1
        User question: "How many rows are in the dataset?"
        Python code:
        answer = len(df)

        Example 2
        User question: "Show me all rows for AAPL"
        Python code:
        result_df = df[df["Stock"] == "AAPL"]
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Here is the actual DataFrame schema:
        {schema_str}

        {examples}

        Now write Python code to answer this user question using df:

        "{question}"

        Remember:
        - Only use df, pd, np.
        - No imports, no file or network access.
        - End by defining either `answer` or `result_df`.
        - Return ONLY the Python code, with no explanations.
        """
    ).strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" if you prefer
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    code = response.choices[0].message.content or ""
    code = clean_code_block(code)
    return code


# =============================
# 4. Very basic safety filter
# =============================
def is_code_safe(code: str) -> bool:
    """
    Naive safety checks to avoid obviously dangerous code.
    This is NOT bullet-proof; consider using AST-based sandboxing in production.
    """
    forbidden = [
        "import ",
        "open(",
        "exec(",
        "eval(",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "pickle",
        "shutil",
        "Popen",
        "input(",
    ]
    lowered = code.lower()
    return not any(token in lowered for token in forbidden)


# =============================
# 5. Streamlit UI
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
