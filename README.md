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
