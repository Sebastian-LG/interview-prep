import azure.functions as func
import datetime
import json
import logging
import pandas as pd
import io
import os
import requests
import base64

app = func.FunctionApp()

@app.route(route="HttpExample", auth_level=func.AuthLevel.FUNCTION)
def hello_world(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing request...")

    try:
        # --- 1. Read parameters ---
        start_date = req.form.get("start_date")
        end_date = req.form.get("end_date")
        count1 = req.form.get("count1")
        count2 = req.form.get("count2")
        category = req.form.get("category")

        # Validate
        if not all([start_date, end_date, count1, count2, category]):
            return func.HttpResponse(
                "Missing one or more required parameters",
                status_code=400
            )

        # Convert integers
        count1 = int(count1)
        count2 = int(count2)

        # --- 2. Read uploaded CSV file ---
        file = req.files.get("datafile")
        if file is None:
            return func.HttpResponse("Missing CSV file", status_code=400)

        content = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        # Optional: validate 3 columns
        if df.shape[1] != 3:
            return func.HttpResponse(
                f"Expected 3 columns in CSV, got {df.shape[1]}",
                status_code=400
            )

        logging.info(f"Received DataFrame with {df.shape[0]} rows and {df.shape[1]} cols")

        # --- 3. Do something with params + file ---
        # For now, just echo back
        response_message = (
            f"Received:\n"
            f"start_date={start_date}, end_date={end_date}, "
            f"count1={count1}, count2={count2}, category={category}\n\n"
            f"CSV Preview:\n{df.head().to_string(index=False)}"
        )

        return func.HttpResponse(response_message, status_code=200)

    except Exception as e:
        logging.error(f"Error: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)
    


# @app.route(route="databricks_trigger", methods=["POST"])
# def trigger(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         body = req.get_json()
#         datos_formulario = body.get("datos_formulario")

#         # Prepare params for Databricks
#         job_params = {
#             "datos_formulario": json.dumps(datos_formulario)  # send as string
#         }

#         # Call Databricks Job
#         databricks_url = f"{os.getenv('DATABRICKS_HOST')}/api/2.1/jobs/run-now"
#         token = os.getenv("DATABRICKS_TOKEN")

#         headers = {"Authorization": f"Bearer {token}"}
#         payload = {
#             "job_id": os.getenv("DATABRICKS_JOB_ID"),
#             "notebook_params": job_params
#         }

#         r = requests.post(databricks_url, headers=headers, json=payload)
#         r.raise_for_status()

#         return func.HttpResponse(
#             f"✅ Triggered Databricks Job: {r.text}",
#             status_code=200
#         )

#     except Exception as e:
#         return func.HttpResponse(f"❌ Error: {str(e)}", status_code=500)
    


@app.route(route="databricks_trigger", methods=["POST"])
def trigger(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # ✅ Get JSON part
        datos_formulario_raw = req.form.get("datos_formulario")
        if not datos_formulario_raw:
            return func.HttpResponse("Missing datos_formulario", status_code=400)

        datos_formulario = json.loads(datos_formulario_raw)

        # ✅ Get file part (optional)
        file = req.files.get("input_file")
        file_b64 = None
        filename = None
        if file:
            file_bytes = file.stream.read()
            file_b64 = base64.b64encode(file_bytes).decode("utf-8")
            filename = file.filename

        # ✅ Prepare params for Databricks
        job_params = {
            "datos_formulario": json.dumps(datos_formulario)
        }

        if file_b64:
            job_params["uploaded_file_name"] = filename
            job_params["uploaded_file_b64"] = file_b64

        # ✅ Call Databricks Job
        databricks_url = f"{os.getenv('DATABRICKS_HOST')}/api/2.1/jobs/run-now"
        token = os.getenv("DATABRICKS_TOKEN")

        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "job_id": os.getenv("DATABRICKS_JOB_ID"),
            "notebook_params": job_params
        }

        r = requests.post(databricks_url, headers=headers, json=payload)
        r.raise_for_status()

        return func.HttpResponse(
            f"✅ Triggered Databricks Job: {r.text}",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(f"❌ Error: {str(e)}", status_code=500)