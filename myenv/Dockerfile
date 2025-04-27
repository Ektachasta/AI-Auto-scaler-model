# 1. Use official Python image
FROM python:3.12-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy all files into the container
COPY . .

# 4. Install the required Python packages
RUN pip install --no-cache-dir dash pandas plotly

# 5. Expose the port (Dash app runs on 8050)
EXPOSE 8050

# 6. Command to run the app
CMD ["python", "app.py"]

