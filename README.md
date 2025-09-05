# 🛰️ Methane Leak Notifier

**Identify methane leaks from satellite data. Notify responsible companies and local authorities automatically.**

This project aims to turn satellite-detected methane emissions into real-world accountability. Using open datasets such as [methanedata.unep.org](https://methanedata.unep.org/map), we analyze methane plumes, identify the most probable responsible companies, and trigger automatic notifications to those organizations and relevant civil authorities.

---

## 🌍 Why This Matters

Methane is over **80 times more potent than CO₂** as a greenhouse gas in the short term. Timely action on methane leaks is one of the most effective ways to slow climate change.

Yet many leaks go unnoticed or unaddressed for weeks—or even months. Our goal is to speed up that feedback loop and make remediation easier, cheaper, and faster.

---

## 🧠 What This Project Does

- Pulls methane emission data from satellite datasets (NASA, UNEP, etc.)
- Matches emissions to nearby industrial facilities using geospatial data
- Uses AI agents to search for company contact details and identify operators
- Sends notifications to:
  - The company or operator
  - Local civil authorities
  - Environmental watchdogs (optional)

---

## ⚙️ Technologies Used

- **Python**
- **AWS** (S3, Lambda, Step Functions, etc.)
- **PostGIS** for spatial queries
- **LangChain + OpenAI** for agent reasoning
- **OpenStreetMap / Wikidata / Business directories** for facility info

---

## 🛠️ Project Structure
- /data/             # Raw data sources, e.g. JSON from UNEP/NASA
- /agents/           # LLM-powered agents to find contact info, analyze responsibility
- /notifications/    # Email, webhook, or API-based messaging logic
- /aws/              # Terraform/CDK configs for cloud deployment
- /frontend/         # (Planned) dashboard to show live methane events

---

## 🚢 Running with Docker

The application now bundles the landing page (`index.html`) and the Streamlit app (`app.py`) in a single container.  
The container uses **nginx** to serve static assets and reverse‑proxy requests to Streamlit.

```bash
# Copy example environment and adjust if needed
cp .env.example .env

# Build and start the stack (app + Postgres)
docker compose up --build

# Visit the landing page
open http://localhost:8508/
# Streamlit app is served under /app/
```

The `docker-compose.yml` file exposes the Postgres service on port `5433` for local development and the application on `8508`.

To run the container without docker‑compose, build and run the image directly:

```bash
docker build -t methane-leaks .
docker run --env-file .env -p 8508:80 methane-leaks
```

---

## ☁️ Deploying to AWS

1. **Build and push the image**
   ```bash
   docker build -t methane-leaks .
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag methane-leaks:latest <account>.dkr.ecr.<region>.amazonaws.com/methane-leaks:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/methane-leaks:latest
   ```
2. **Create an ECS service (Fargate)** using the image above and map port `80` to an Application Load Balancer.
3. **Provide environment variables** for database connectivity (see `.env.example` for required vars).
4. **Attach a database**: use Amazon RDS for PostgreSQL or an external Postgres instance and supply its credentials via the environment.

This setup results in a single publicly accessible endpoint that serves the landing page at `/` and the Streamlit app under `/app/`.

---

## 🧑‍🤝‍🧑 How to Contribute

This is an **open collaboration project** — contributions are welcome!

- 🐛 Report issues or ideas in [Issues](https://github.com/your-org/methane-leak-notifier/issues)
- 🧪 Submit a pull request (see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines)
- 🌍 Help map industrial facilities to methane leaks
- 🧠 Improve agent logic for identifying responsible entities
- 📨 Suggest or improve contact strategies


---

## 📜 License

This project is licensed under the **MIT License**.

---

## ✉️ Contact

For ideas, questions, or partnerships, feel free to open an issue or reach out directly.
