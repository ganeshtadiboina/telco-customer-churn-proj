from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in the telecom industry",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        return {"prediction": predict(payload)}
    except Exception as exc:
        return {"error": str(exc)}


def _build_payload(
    gender,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
):
    return {
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
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }


def gradio_predict(*values):
    try:
        result = predict(_build_payload(*values))
    except Exception as exc:
        return f"<div class='result result-error'>Prediction failed<br><span>{exc}</span></div>"

    result_class = "result-high" if result == "Likely to churn" else "result-low"
    label = "High churn risk" if result == "Likely to churn" else "Low churn risk"
    return f"<div class='result {result_class}'>{label}<span>{result}</span></div>"


def load_high_risk_sample():
    return (
        "Female","No","No","Yes","No","Fiber optic",
        "No","No","No","No","Yes","Yes",
        "Month-to-month","Yes","Electronic check",
        1,85.0,85.0,
    )


def load_low_risk_sample():
    return (
        "Male","Yes","Yes","Yes","Yes","DSL",
        "Yes","Yes","Yes","Yes","No","No",
        "Two year","No","Credit card (automatic)",
        60,45.0,2700.0,
    )


CSS = """
html,
body,
#root,
gradio-app {
    height:100%;
    margin:0;
}

.gradio-container {
    max-width:none !important;
    width:100% !important;
    min-height:100vh !important;
    display:flex !important;
    flex-direction:column !important;
    padding-left:0 !important;
    padding-right:0 !important;
    padding-bottom:0 !important;
}

.app-shell {
    max-width:1180px;
    margin:0 auto;
    padding:48px 0 8px;
}

.main-workspace {
    max-width:1180px;
    margin:0 auto !important;
    padding-bottom:72px;
}

.app-title {
    margin:0;
    font-size:32px;
    font-weight:800;
    color:#18202f;
}

.app-subtitle {
    margin:4px 0 0;
    color:#5b6475;
    font-size:15px;
}

.panel {
    border:1px solid #e1e6ef;
    border-radius:8px;
    padding:14px;
    background:#ffffff;
}

.section-title {
    margin:0 0 10px;
    color:#273244;
    font-size:15px;
    font-weight:700;
}

.form-tabs > .tab-nav {
    margin-bottom:10px;
}

.predict-panel {
    min-height:360px;
}

.result {
    min-height:132px;
    border-radius:8px;
    padding:24px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    border:1px solid #475569 !important;
    color:#f8fafc !important;
    background:#111827 !important;
    font-size:16px;
    font-weight:700;
    box-shadow:0 16px 36px rgba(0,0,0,.22);
}

.result span {
    display:block;
    margin-top:8px;
    font-size:26px;
    line-height:1.2;
    color:#fff !important;
}

.result-high {
    border-color:#fb7185 !important;
    background:#881337 !important;
}

.result-low {
    border-color:#34d399 !important;
    background:#14532d !important;
}

.result-error {
    border-color:#fbbf24 !important;
    background:#78350f !important;
}

.sample-row {
    gap:8px;
}

div:has(> .app-footer) {
    width:100% !important;
    margin-top:auto !important;
    padding:0 !important;
}

.app-footer {
    width:100%;
    min-height:215px;
    margin:0;
    padding:34px 0 44px;
    border-top:1px solid #263244;
    background:#0b1220;
    color:#cbd5e1;
    display:flex;
    align-items:flex-start;
}

.footer-grid {
    display:grid;
    grid-template-columns:1.3fr 1fr 1fr;
    gap:60px;
    width:100%;
    align-items:start;
    max-width:1180px;
    margin:0 auto;
    padding:0;
}

.footer-title {
    margin:0 0 6px;
    color:#fff;
    font-size:15px;
    font-weight:800;
}

.footer-text {
    margin:0;
    color:#cbd5e1;
    font-size:13px;
    line-height:1.55;
}

.footer-pill-row {
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:10px;
}

.footer-pill {
    padding:5px 9px;
    border-radius:999px;
    background:#1f2937;
    color:#e5e7eb;
    border:1px solid #334155;
    font-size:12px;
    font-weight:700;
}

.footer-link {
    color:#93c5fd !important;
    text-decoration:none;
    font-weight:700;
}

@media (max-width:860px){
    .app-shell,
    .main-workspace {
        width:calc(100% - 36px);
    }

    .app-shell {
        padding-top:32px;
    }

    .main-workspace {
        padding-bottom:42px;
    }

    .app-footer {
        min-height:auto;
        padding:28px 0 36px;
    }

    .footer-grid {
        grid-template-columns:1fr;
        gap:22px;
        width:calc(100% - 36px);
        padding:0;
    }
}

footer {
    display:none !important;
}
"""


with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Telco Churn Predictor") as demo:

    gr.HTML('''
    <div class="app-shell">
        <h1 class="app-title">Telco Churn Predictor</h1>
        <p class="app-subtitle">Customer profile, services, billing, and account signals in one compact workspace.</p>
    </div>
    ''')

    with gr.Row(equal_height=True, elem_classes=["main-workspace"]):

        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.HTML("<p class='section-title'>Customer Factors</p>")

            with gr.Tabs(elem_classes=["form-tabs"]):
                with gr.Tab("Profile"):
                    with gr.Row():
                        gender = gr.Dropdown(["Male","Female"], value="Male", label="Gender")
                        Partner = gr.Dropdown(["Yes","No"], value="No", label="Partner")
                        Dependents = gr.Dropdown(["Yes","No"], value="No", label="Dependents")

                    with gr.Row():
                        tenure = gr.Slider(0,100,value=1,step=1,label="Tenure")
                        MonthlyCharges = gr.Slider(0,200,value=85,step=.5,label="Monthly Charges")
                        TotalCharges = gr.Slider(0,10000,value=85,step=1,label="Total Charges")

                with gr.Tab("Services"):
                    with gr.Row():
                        PhoneService=gr.Dropdown(["Yes","No"],value="Yes",label="Phone Service")
                        MultipleLines=gr.Dropdown(["Yes","No","No phone service"],value="No",label="Multiple Lines")
                        InternetService=gr.Dropdown(["DSL","Fiber optic","No"],value="Fiber optic",label="Internet Service")

                    with gr.Row():
                        OnlineSecurity=gr.Dropdown(["Yes","No","No internet service"],value="No",label="Online Security")
                        OnlineBackup=gr.Dropdown(["Yes","No","No internet service"],value="No",label="Online Backup")
                        DeviceProtection=gr.Dropdown(["Yes","No","No internet service"],value="No",label="Device Protection")

                    with gr.Row():
                        TechSupport=gr.Dropdown(["Yes","No","No internet service"],value="No",label="Tech Support")
                        StreamingTV=gr.Dropdown(["Yes","No","No internet service"],value="Yes",label="Streaming TV")
                        StreamingMovies=gr.Dropdown(["Yes","No","No internet service"],value="Yes",label="Streaming Movies")

                with gr.Tab("Billing"):
                    Contract=gr.Dropdown(
                        ["Month-to-month","One year","Two year"],
                        value="Month-to-month",
                        label="Contract"
                    )

                    with gr.Row():
                        PaperlessBilling=gr.Dropdown(["Yes","No"],value="Yes",label="Paperless Billing")
                        PaymentMethod=gr.Dropdown(
                            [
                                "Electronic check",
                                "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)"
                            ],
                            value="Electronic check",
                            label="Payment Method"
                        )

        with gr.Column(scale=4, elem_classes=["panel","predict-panel"]):
            gr.HTML("<p class='section-title'>Prediction</p>")

            output = gr.HTML(
                "<div class='result'>Ready<span>Run prediction</span></div>"
            )

            predict_button = gr.Button("Predict Churn", variant="primary")

            with gr.Row(elem_classes=["sample-row"]):
                high_sample = gr.Button("High Risk Sample")
                low_sample = gr.Button("Low Risk Sample")

    inputs=[
        gender,Partner,Dependents,
        PhoneService,MultipleLines,InternetService,
        OnlineSecurity,OnlineBackup,DeviceProtection,
        TechSupport,StreamingTV,StreamingMovies,
        Contract,PaperlessBilling,PaymentMethod,
        tenure,MonthlyCharges,TotalCharges
    ]

    predict_button.click(gradio_predict, inputs=inputs, outputs=output)
    high_sample.click(load_high_risk_sample, outputs=inputs)
    low_sample.click(load_low_risk_sample, outputs=inputs)

    gr.HTML('''
    <div class="app-footer">
      <div class="footer-grid">
        <div>
          <p class="footer-title">Telco Churn Predictor</p>
          <p class="footer-text">
            A compact ML workspace for estimating customer churn risk from service,
            billing, and account behavior signals.
          </p>
          <div class="footer-pill-row">
            <span class="footer-pill">XGBoost</span>
            <span class="footer-pill">MLflow</span>
            <span class="footer-pill">FastAPI</span>
            <span class="footer-pill">Gradio</span>
          </div>
        </div>

        <div>
          <p class="footer-title">Model Inputs</p>
          <p class="footer-text">
             Profile, services, contracts, payment methods,
             tenure, monthly charges and total charges.
          </p>
        </div>

        <div>
          <p class="footer-title">Developer Access</p>
          <p class="footer-text">
            REST endpoint: <span class="footer-pill">POST /predict</span><br>
            API docs: <a class="footer-link" href="/docs" target="_blank">/docs</a>
          </p>
        </div>
      </div>
    </div>
    ''')


app = gr.mount_gradio_app(app, demo, path="/ui")
