# Nutritional AI - Calorie and Diet Plan Assistant

## Overview
This project includes a backend and frontend for a **calorie and diet plan assistant**.  
The backend is built using **FastAPI**, **Poetry**, **Langgraph**, and **LangChain**, while the frontend is built with **Next.js**, **TypeScript**, and **Tailwind CSS**.

- The **backend** calculates daily caloric needs based on the user's data and provides **personalized diet plans**.
- The **frontend** is a **responsive web app** that interacts with the backend API, providing users with a seamless experience to input their information and view their recommended calorie intake and diet plans.

The system can:
- **Book doctor appointments** using specific functions.
- **Calculate daily calories** and **recommend diet plans** according to user needs.

---

## Table of Contents

1. [Backend (FastAPI with Poetry)](#backend-fastapi-with-poetry)
   - [Requirements](#requirements)
   - [Setup](#setup)
   - [API Documentation](#api-documentation)
   - [Running the Backend](#running-the-backend)
2. [Frontend (Next.js with TypeScript and Tailwind CSS)](#frontend-nextjs-with-typescript-and-tailwind-css)
   - [Requirements](#requirements-1)
   - [Running the Frontend](#running-the-frontend)
3. [Project Structure](#project-structure)
4. [Environment Variables](#environment-variables)

---

## Backend (FastAPI with Poetry)

### Requirements

- Python 3.x
- **Poetry** (for dependency management)
- FastAPI
- Uvicorn
- LangChain
- Langgraph
- Google Generative AI
- Tavily API
- FAISS
- dotenv
- pydantic
- faiss-cpu
- beautifulsoup4

---

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ehtasham11/nutritional-AI.git
   cd nutritional-AI
   ```

2. **Create a `.env` file in the backend directory and add the following environment variables:**
   ```
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

3. The backend uses:
   - Google Generative AI API
   - Tavily API for search
   - FAISS for vector-based searches

---

## Running the Backend
Run the following command inside the backend directory:
```bash
poetry run python -m uvicorn main:app --port 8001 --reload
```

---

## Frontend (Next.js with TypeScript and Tailwind CSS)

### Requirements

- Node.js (v14 or higher)
- npm
- Tailwind CSS
- TypeScript

---

### Running the Frontend
Run the following command inside the frontend directory:
```bash
npm run dev
```

---

## Project Structure

```
├── backend/
│   ├── main.py             # FastAPI application entry point
│   ├── pyproject.toml      # Poetry configuration
│   ├── .env                # Environment variables
├── frontend/
│   ├── app/
│   │   ├── globals.css     # Global styles
│   │   ├── layout.tsx      # Layout component
│   │   ├── page.tsx        # Home page (input forms, output display)
│   ├── public/             # Public assets (images, icons, etc.)
│   ├── tsconfig.json       # TypeScript configuration
│   ├── package.json        # Frontend dependencies and scripts
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

---

## Environment Variables

Backend:
- `GOOGLE_API_KEY`: API key for Google Generative AI
- `TAVILY_API_KEY`: API key for Tavily search

Frontend:
- `API_URL`: Base URL for the backend API (default: `http://localhost:8001`)

**Important:**  
Make sure to configure the `.env` files for both backend and frontend accordingly.

---

## Features

- **Doctor Appointment Booking:**  
  Allows users to easily book a doctor appointment through a specific backend function.

- **Calorie Calculation and Diet Plan:**  
  Calculates how many calories a user needs daily based on their:
  - Age
  - Weight
  - Height
  - Gender
  - Activity level  
  Then, it generates a **personalized diet plan**.

---

