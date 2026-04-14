# Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean


# Create the target directory
RUN mkdir -p /home/root/data_grid2op/
# Clone the repository
RUN git clone https://github.com/AI4REALNET/grid2op-scenario.git /tmp/grid2op-scenario
# Copy the folder into the data directory
RUN mkdir -p /root/data_grid2op
RUN cp -r /tmp/grid2op-scenario/ai4realnet_small /root/data_grid2op/ai4realnet_small
# Cleanup
RUN rm -rf /tmp/grid2op-scenario

RUN git clone https://github.com/Mleyliabadi/ExpertOp4Grid.git /tmp/ExpertOp4Grid
RUN pip install /tmp/ExpertOp4Grid/.
RUN rm -rf /tmp/ExpertOp4Grid


# Install OS deps (if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc && \
#     rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy API + agent package
COPY app ./app/
COPY ExpertAgent ./ExpertAgent/
COPY PPO_SB3 ./PPO_SB3/
COPY setup.py .
COPY README.md .
# Copy the whole project
# COPY . .


# Install the project using setup.py at the root
RUN pip install .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
