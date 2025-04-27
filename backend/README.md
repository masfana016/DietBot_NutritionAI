# Doctor Appointment and Nutrition Assistant

## Overview

This project is a FastAPI-based backend offering two major services:

- **Doctor Appointment Booking System**
- **Nutrition, Calorie, and Diet AI Assistant**

The system uses AI models, local database storage, and external search tools to assist users with healthcare and fitness needs.

---

## Features

### 1. Doctor Appointment Booking

- Allows users to book an appointment with a doctor easily.
- Uses a database model (`Appointment`) to store:
  - Doctor's Name
  - Specialization
  - Appointment Date
  - Appointment Time
- The `book_appointment` function saves the appointment information into a local SQLite database and provides a confirmation.

### 2. Nutrition, Calorie, and Diet Assistant

- Uses **Google Gemini AI model** to answer health, fitness, and diet-related queries.
- Implements a **Retrieval-Augmented Generation (RAG)** system:
  - Fetches relevant nutrition information from local documents and online articles.
- **Calorie Calculation Tool**:
  - Calculates the number of calories a person should consume daily based on:
    - Gender
    - Age
    - Weight
    - Height
    - Activity Level
  - Helps users understand **how much they need to eat** depending on whether they want to **maintain**, **gain**, or **lose** weight.
- **Diet Plan Generator**:
  - Based on calorie needs, it suggests a personalized daily diet plan.

---

## Technologies Used

- **FastAPI** - API backend
- **Google Generative AI** - Conversational AI
- **FAISS** - Smart document search with embeddings
- **SQLModel** and **SQLite** - Database for appointments
- **LangChain** - AI tools and agent orchestration
- **Tavily API** - Web search for health information
- **Pydantic** - Data validation
- **CORS Middleware** - Allow frontend-backend interaction

---

## How it Works

- The system provides an API where users can:
  - **Book a doctor's appointment** by providing relevant details.
  - **Ask health or nutrition-related questions**, and get answers.
  - **Request a calorie calculation**, and the system will:
    - Collect basic personal info.
    - Tell them how many calories they should eat daily.
    - Provide a customized diet plan based on their goals (gain, lose, or maintain weight).

Tools and functions are intelligently selected according to the type of query.

---

## Setup Instructions

1. Clone the repository.
2. Install all dependencies using Poetry:
   ```
   poetry install
   ```
3. Run the Fastapi server:
   ```
   poetry run python -m uvicorn main:app --port 8001 --reload
   ```
4. Open the API docs at:
   ```
   http://localhost:8000/docs
   ```

---

## Future Improvements

- Add user authentication for appointment security.
- Include food preferences in diet plans (e.g., halal, vegetarian).
- Expand doctor profiles with ratings and feedback options.

---

## License

This project is licensed under the [MIT License](LICENSE).
