
# Create the base image
FROM python:3.7-slim

# Change the working directory
WORKDIR /app/

# Install Dependency
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

# Copy local folder into the container
COPY app.py /app/
COPY xgboost_gender_jupyter.pkl /app/
COPY xgboost_age-jupyter.pkl /app/
COPY df_test.pkl /app/
COPY events_active_labels_encoder.pkl /app/
COPY events_brand_encoder.pkl /app/
COPY events_installed_labels_encoder.pkl /app/
COPY events_model_encoder.pkl /app/
COPY templates/view.html /app/templates/view.html
COPY static/style.css /app/static/style.css


# Set "python" as the entry point
ENTRYPOINT ["python"]

# Set the command as the script name
CMD ["app.py"]

#Expose the post 5000.
EXPOSE 5000