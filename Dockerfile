FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Expose Streamlit port
EXPOSE 10000

# Run Streamlit on proper port/address
CMD ["streamlit", "run", "finance_wizard_dashboard.py", "--server.port=10000", "--server.address=0.0.0.0"]


