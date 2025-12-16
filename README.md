# 🌙 Lunar Crater Age Classifier API

This repository contains the Machine Learning prediction service, built with **FastAPI** and **TensorFlow/Keras**. It processes uploaded lunar image chipouts, classifies them into age categories (New Crater, Old Crater, No Crater), and provides an **Explainable AI (XAI)** heatmap using Grad-CAM.

---

## 💻 Local Development Setup

These instructions guide you through getting the API running on your local machine.

### Prerequisites

1.  **Python 3.10+**
2.  **`make`** utility
3.  **Docker** (for containerized setup)
4.  **Google Cloud SDK** (for deployment)

### ⚙️ Quick Start using `make`

The project uses a `Makefile` to simplify common tasks.

1.  **Install Dependencies:**
    ```bash
    make install
    ```

2.  **Run the API Locally:**
    This command starts the Uvicorn server with hot-reloading on `http://127.0.0.1:8000`.
    ```bash
    make api
    ```
    This automatically opens the interactive API documentation (Swagger UI) in your browser.

### 📜 Available `make` Commands

| Command | Description |
| :--- | :--- |
| `make install` | Installs all dependencies from `requirements.txt` and updates the local package setup. |
| `make clean` | Removes all temporary files, build artifacts (`__pycache__`, `.coverage`, `dist`), and old metadata. |
| `make test_structure` | Runs the file structure check script. |
| `make run-api` | Starts the FastAPI service on `localhost:8000` (without opening docs). |
| `make api` | Runs the API service in the background and opens the `/docs` page. |

---

## 🐳 Dockerized Environment

The API can be run locally using Docker, mimicking the production environment.

### Local Docker Commands

| Command | Description |
| :--- | :--- |
| `make docker_build_local` | Builds the Docker image locally using your local architecture (`linux/amd64` or `linux/arm64`), tagged as `:local`. |
| `make docker_run_local` | Runs the local Docker image on port `8000`. Requires a `.env` file for configuration. |
| `make docker_run_local_interactively` | Runs the local image and opens an interactive `bash` shell inside the running container (useful for debugging). |

---

## ☁️ Deployment to Google Cloud Platform (GCP)

The API is deployed as a Docker container to **Google Cloud Run** via **Google Artifact Registry (GAR)**.

### Prerequisites (GCP)

Ensure the following variables are set in your shell environment or a dedicated configuration file: `GCP_PROJECT`, `GCP_REGION`, `DOCKER_REPO_NAME`, `DOCKER_IMAGE_NAME`, `GAR_MEMORY`, and `GCP_SERVICE_ACCOUNT`.

### Deployment Steps

1.  **Set GCP Project:**
    ```bash
    make gcloud-set-project
    ```

2.  **Authenticate Docker and Create Repository (First time only):**
    ```bash
    make docker_allow
    make docker_create_repo
    ```

3.  **Build Production Image (`linux/amd64`):**
    This ensures compatibility with Cloud Run.
    ```bash
    make docker_build
    ```

4.  **Push and Deploy:**
    ```bash
    # Push the :prod tagged image to Artifact Registry
    make docker_push

    # Deploy the latest image to the Cloud Run service
    make docker_deploy
    ```

### GCP Utility Commands

| Command | Description |
| :--- | :--- |
| `make docker_show_image_path` | Prints the full, canonical path of the image in the Artifact Registry. |
| `make docker_run` | Runs the cloud-built (`linux/amd64`) image locally for pre-deployment testing. |
