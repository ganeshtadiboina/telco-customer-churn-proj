"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving 
===========================================================================

This application provides a complete serving solutions fo rthe Telco Customer model with both programmatic API access and user-friendly web interface.

Architecture:
- FastAPI : High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstartions
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict # Core ML inferecne logic

# Initialize FastAPI application
app = FastAPI(
  title="Telco Customer Churn Prediction API",
  description="ML API for predicting customer churn in telecom industry",
  version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
  """
  Health check endpoint for monitoring and load balancer health checks.
  """

  return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
  """

  Customer data schema for churn prediction.

  This schema defines the exact 18 features required for churn prediction.

  All features match the original dataset structure for consistency.
  """

  # Demographics
  gender: str  # "Male" or "Female"
  Partner: str # "yes" or "No" - has partner
  Dependents: str # "Yes" or "No" - has dependents

  # Phone Services
  PhoneService: str # "Yes" or "No"
  MultipleLines: str # "Yes", "No", or "No phone service"

  # Internet services
  InternetService: str # "DSL", "Fiber Optic", or "No"
  OnlineSecurity: str # "Yes", "No", or "No internet service"
  OnlineBackup: str # "Yes", "No", or "No internet service"
  DeviceProtection: str # "Yes", "No" or "No"
  TechSupport: str # "Yes", "No", or "No internet service"
  StreamingMovies: str # "Yes", "No", or "No internet service"

  # Account information
  Contract: str # "Month-to-month","One year", "Two year"
  PaperlessBilling: str # "Yes" or "No"
  PaymentMethod: str # "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

  # Numeric features
  tenure: int # Number of months the customer has been with the company
  MonthlyCharges: float # Monthly charges in dollars
  TotalCharges: float # Total charges to date

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
  """
  Main prediction endpoint for customer churn prediction.

  This endpoint:
  1. Recieves validated customer data via Pydantic model 
  2. Calls the inference pipeline to transform features and predict
  3. Returns churn prediction in JSON format

  Expected Response:
  - {"prediction": "Likely to churn" or "Not likely to churn"}
  - {"error": "error_message"} if prediction fails
  """
  try:
    # Convert Pydantic model to dict and call inference pipeline
    result = predict(data.dict())
    return {"prediction": result}
  except Exception as e:
    # Return error details for debugging (consider logging in production)
    return {"error": str(e)}

# ========================================================================== #

# === GRADIO INTERFACE ===

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
  """
  Gradio interface function that processes from inputs and returns prediction.

  This function:
  1. Takes individual form inputs from Gradio UI
  2. Constructs the data dictionary matching the API schema
  3. Calls the same inference pipeline used by the API 
  4. Returns user-friendly prediction string

  """
  # Construct data dictionary matching CustomerData
  data = {
    "gender": gender,
    "Partner": Partner,
    "Dependents": Dependents,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "tenure": int(tenure),  #Ensure integer type
    "MonthlyCharges": float(MonthlyCharges), #Ensure float type
    "TotalCharges": float(TotalCharges) #Ensure float type
  }

  try:
    # Call the same inference pipeline used by the API
    result = predict(data)
    return str(result) # Return as string for Gradio display
  except Exception as e:
    return f"Prediction failed: {str(e)}"

# ========================================================================== #

# === GRADIO UI CONFIGURATION ===  
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
  fn=gradio_interface,
  inputs=[
    #Demographics section
    gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
    gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
    gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

    # Phone Services section
    gr.Dropdown(["Yes", "No"], lable="Phone Service", value="Yes"),
    gr.Dropdown(["Yes", "No", "No phone service"], lable="Multiple Lines", value="No"),

    # Internet Services section (key churn predictors)
    gr.Dropdown(["DSL", "Fiber optic", "No"]),
    gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
    gr.Dropdown(["Yes", "No", "No internet service"], lable="Device Protection", value="No"),
    gr.Dropdown(["Yes", "No", "No internet service"], lable="Tech Support", value="No"),
    gr.Dropdown(["Yes", "No", "No internet service"], lable="Streaming TV", value="Yes"),
    gr.Dropdown(["Yes", "No", "No internet service"], lable="Streaming Movies", value="Yes"),


    # Contract and billing section (major churn factors)
    gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"), 
    gr.Dropdown(["Yes", "No"], lable="Paperless Billing", value="Yes"),
    gr.Dropdown([
      "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card(automatic)"
    ], lable="Payment Method", value="Electronic check"),


    # Numeric features (important for churn prediction)
    gr.Number(lable="Tenure (months)", value=1, minimum=0, maximum=100),
    gr.Number(lable="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
    gr.Number(lable="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
  ],
  outputs=gr.Textbox(lable="Churn Prediction", lines=2),
  title="Telco Customer Churn Predictor",
  description="""
  **Predict customer churn probability using machine learning**

  Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on histomrical telecom customer data to identify customers at risk of churning.

  **Tip**: Month-to-month contracts with fiber optic internet and electronic check payments tent to have higher churn rates.
  """,
  examples=[
    #High churn risk example
    ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
    1, 85.0, 85.0],
    # Low churn risk example
    ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes","Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0]
  ],
  theme=gr.themes.Soft() # Professional appearance
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
  app, # FastAPI application instance
  demo, # Gradio interface
  path="/ui" # URL path where Gradio will be accessible
)