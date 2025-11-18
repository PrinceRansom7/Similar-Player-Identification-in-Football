# ⚽ Similar Player Identification System

### AI-Powered Football Player Recommendation & Comparison Engine

This project is a full-stack football scouting tool that identifies similar players, compares them visually & statistically, and generates AI-powered scouting reports using **Google Gemini**.

Built with:

- **FastAPI** (Backend API)
- **Vanilla JavaScript + HTML + CSS** (Frontend)
- **Chart.js Radar Charts**
- **Gemini Pro** (AI player comparison)
- **NumPy + Pandas + Scikit-Learn** (Similarity Engine)

# ⭐ Features

### 🔍 Player Search

Search players instantly using substring or partial name matching.

### 📊 Similar Player Recommendation

Given a player, find **k most similar players** based on normalized feature vectors using **cosine similarity**.

### 📈 Radar Chart Visualisation

- Displays multi-attribute player profiles
- Highlights input player permanently
- Clicking a similar player:
  - Pulsates their radar polygon
  - Fades all other comparison polygons (Option C)
- Smooth animations
- Clean white radar on dark theme UI

### 👥 Multi-Player Comparison

- Select **multiple players** via checkboxes
- View their stats in horizontally scrollable boxes
- Compare their radar overlays
- Toggle between:
  - Position-based feature set
  - Full feature set

### 🤖 AI Scouting Report (Gemini Pro)

Generates:

- Tactical profile
- Strengths
- Weaknesses
- Role fit
- Direct comparisons between selected players

### 🧰 Live Tooltips

- Hovering radar points shows:
  - **Feature name**
  - **Feature value**
  - **Feature description**
- Hovering stats shows descriptions of each metric

### 🎚 Smart Filters

Filter similar players by:

- League
- Position
- Age range
- Nationality

---

Made with ❤️ for football analytics
