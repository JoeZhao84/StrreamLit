from textwrap import dedent

import pandas as pd
from openai import OpenAI

# =============================
# OpenAI client
# =============================
# Make sure OPENAI_API_KEY is set in your environment or Streamlit secrets.
client = OpenAI()  # reads OPENAI_API_KEY by default


# =============================
# Helper: clean fenced code
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
# Code generator using ChatGPT
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


