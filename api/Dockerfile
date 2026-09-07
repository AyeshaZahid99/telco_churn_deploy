# Python version is NOT arbitrary here — see the note at the top of main.py.
# The pickled pipeline was created under Python 3.10.5 and must be loaded
# under Python 3.10.x or predictions will be silently wrong.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pipeline.pkl my_feature_dict.pkl ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
