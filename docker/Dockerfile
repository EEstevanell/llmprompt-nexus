FROM python:3.9.13

WORKDIR /llmprompt-nexus

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .
RUN curl -sSL https://install.python-poetry.org | python3 - && export PATH="$HOME/.local/bin:$PATH" && poetry config virtualenvs.create false && poetry install

RUN pip install -e /llmprompt-nexus

# Set environment variables
ENV PYTHONPATH=/llmprompt-nexus