# Use the official DOLFINx Docker image as the base
FROM dolfinx/dolfinx:stable

# Copy the requirements file to the container
COPY requirements.txt /workspace/requirements.txt

# Install any additional Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy scipy matplotlib jupyterlab  # Add other packages as needed

RUN python3 -m pip install scifem

# Expose Jupyter port
EXPOSE 8888

# Set default command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

