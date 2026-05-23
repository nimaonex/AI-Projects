import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from langchain_openai.chat_models import ChatOpenAI

# ---------------------------------------------------------
# 1. AGENT: Natural Language Explanation
# ---------------------------------------------------------

def explain_results(pred_value, mse, r2):
    """
    Uses an LLM (Agent) to explain model results in simple language.
    """

    prompt = f"""
    You are a helpful data science assistant.
    Explain the following linear regression results in very simple terms
    so that a non-technical person can understand:

    - Predicted value: {pred_value}
    - Model Mean Squared Error: {mse}
    - R² score: {r2}
    In this dataset each column is represented as below:
    - maindate: month of the year
    - NetQTY: total number of sales
    - TAX: tax 
    - Discount: discount
    - customer_count: number of customers
    - product_count: number of sold products or total number of SKU
    - NetAmount: is the target for model and refer to total sales amount

    Provide a clear explanation of:
    * What the model is trying to predict
    * What the prediction means
    * Whether the model seems accurate
    * Any advice for improving data or model
    """

    llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="qwen2.5-3b-instruct",
    temperature=0,
    max_completion_tokens=1000
)

    return llm.invoke(prompt).content


# ---------------------------------------------------------
# 2. BUILD AGENTIC ML APP
# ---------------------------------------------------------
def run_agentic_linear_regression(file_path, target_column):

    # Load Excel dataset
    df = pd.read_excel(file_path)

    # Split into X (features) and y (target)
    #X = df.drop(columns=[target_column])
    #y = df[target_column]
    X = df.iloc[:, 2:-1]
    y = df.iloc[:, -1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make a prediction for the first test row
    first_sample = X_test.iloc[0:1]
    prediction = model.predict(first_sample)[0]

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Ask LLM to interpret the results
    explanation = explain_results(prediction, mse, r2)

    return {
        "sample_input": first_sample.to_dict(orient="records")[0],
        "prediction": prediction,
        "mse": mse,
        "r2_score": r2,
        "llm_explanation": explanation
    }


# ---------------------------------------------------------
# 3. RUN THE APPLICATION
# ---------------------------------------------------------
if __name__ == "__main__":
    result = run_agentic_linear_regression(
        file_path=r"E:\AI Projects\My Project\Files\aggr.xlsx",
        target_column="NetAmount"
    )

    print("\n=== INPUT SAMPLE ===")
    print(result["sample_input"])

    print("\n=== PREDICTION ===")
    print(result["prediction"])

    print("\n=== MODEL METRICS ===")
    print("MSE:", result["mse"])
    print("R²:", result["r2_score"])

    print("\n=== AGENT EXPLANATION ===")
    print(result["llm_explanation"])
