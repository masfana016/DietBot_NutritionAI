from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain.tools import tool
from langchain.tools import Tool
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

app = FastAPI()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             google_api_key=os.getenv("GOOGLE_API_KEY"))

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

loader1 = WebBaseLoader("https://www.healthline.com/nutrition/1500-calorie-diet#foods-to-eat")
loader2 = WebBaseLoader("https://www.msdmanuals.com/home")
loader3 = WebBaseLoader("https://www.eatingwell.com/category/4305/weight-loss-meal-plans/")
docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()
combined_docs = docs1 + docs2 + docs3
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(combined_docs)
vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "healthline_search",
    "Search for information about healthline, food and nutrition. For any questions about food and nutrition, healthy diet related and just answer the question, don't explain much, you must use this tool!",
)

@tool("calorie_calculator")
def calorie_calculator_tool(input_text: str):
    """
    Calculate the estimated daily calorie needs based on user inputs such as weight, height, age, gender, and activity level,
    and generate a basic diet plan to meet these calorie requirements with balanced meals.
    Also, provide guidance if the user needs to gain or lose weight based on calorie needs, age, and gender.
    """
    try:
        params = {}
        for param in input_text.split(','):
            key, value = param.split('is')
            params[key.strip()] = value.strip()

        weight = float(params.get('weight', 70)) 
        height = float(params.get('height', 175))
        age = int(params.get('age', 25))
        gender = params.get('gender', 'male').lower()
        activity_level = params.get('activity_level', 'moderate').lower()

        # Calculate daily calorie needs based on user gender and activity level
        if gender == "male":
            calories = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        elif gender == "female":
            calories = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            return {"response": "Gender must be either 'male' or 'female'."}

        calories *= {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725
        }.get(activity_level, 1.55)

        # Define base_calories adjustment logic
        base_calories = calories  # Initialize with the calculated calories

        if gender == "female":
            if activity_level == "light":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, calories), 1400)
                elif 7 <= age <= 18:
                    base_calories = min(max(1200, calories), 1800)
                elif 19 <= age <= 60:
                    base_calories = min(max(1600, calories), 2000)
                elif age >= 61:
                    base_calories = max(1600, calories)
            elif activity_level == "active":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, calories), 1600)
                elif 7 <= age <= 18:
                    base_calories = min(max(1600, calories), 2400)
                elif 19 <= age <= 60:
                    base_calories = min(max(1800, calories), 2400)
                elif age >= 61:
                    base_calories = min(max(1800, calories), 2000)

        elif gender == "male":
            if activity_level == "light":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, calories), 1400)
                elif 7 <= age <= 18:
                    base_calories = min(max(1400, calories), 2400)
                elif 19 <= age <= 60:
                    base_calories = min(max(2200, calories), 2600)
                elif age >= 61:
                    base_calories = max(2000, calories)
            elif activity_level == "active":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, calories), 1800)
                elif 7 <= age <= 18:
                    base_calories = min(max(1600, calories), 3200)
                elif 19 <= age <= 60:
                    base_calories = min(max(2400, calories), 3000)
                elif age >= 61:
                    base_calories = min(max(2200, calories), 2600)

        # Determine if the user needs to gain, lose, or maintain weight
        if calories < base_calories:
            weight_goal = "You should increase your calorie intake to gain weight."
        elif calories > base_calories:
            weight_goal = "You should decrease your calorie intake to lose weight."
        else:
            weight_goal = "Your calorie intake is appropriate for maintaining your current weight."

        # Define diet plans
        diet_plans = {
            "1500": {
                "goal": "Weight Loss",
                "plan": [
                    "Breakfast: Greek yogurt with berries and chia seeds",
                    "Morning Snack: Apple with almond butter",
                    "Lunch: Grilled chicken salad with greens and vinaigrette",
                    "Afternoon Snack: Cottage cheese with cucumber",
                    "Dinner: Baked salmon, asparagus, and quinoa",
                    "Evening Snack: Almonds"
                ]
            },
            "1800": {
                "goal": "Weight Maintenance",
                "plan": [
                    "Breakfast: Scrambled eggs with spinach, whole-grain toast",
                    "Morning Snack: Orange and walnuts",
                    "Lunch: Turkey wrap with hummus and veggies",
                    "Afternoon Snack: Banana with peanut butter",
                    "Dinner: Grilled chicken, sweet potato, and broccoli",
                    "Evening Snack: Cottage cheese with berries"
                ]
            },
            "2000": {
                "goal": "Moderate Weight Gain",
                "plan": [
                    "Breakfast: Overnight oats with banana and peanut butter",
                    "Morning Snack: Smoothie with protein powder",
                    "Lunch: Brown rice bowl with black beans and salsa",
                    "Afternoon Snack: Toast with cottage cheese and tomatoes",
                    "Dinner: Steak, mashed potatoes, and green beans",
                    "Evening Snack: Greek yogurt with honey and pumpkin seeds"
                ]
            },
            "2200": {
                "goal": "Active Weight Maintenance/Gain",
                "plan": [
                    "Breakfast: Omelet with veggies and whole-grain toast",
                    "Morning Snack: Greek yogurt with granola and blueberries",
                    "Lunch: Tuna wrap with veggies",
                    "Afternoon Snack: Apple and almonds",
                    "Dinner: Roasted chicken, brown rice, and carrots",
                    "Evening Snack: Dark chocolate with walnuts"
                ]
            },
            "2500": {
                "goal": "High-Calorie for Weight Gain",
                "plan": [
                    "Breakfast: Smoothie bowl with peanut butter and granola",
                    "Morning Snack: Crackers with cheese and apple",
                    "Lunch: Quinoa bowl with chickpeas and roasted veggies",
                    "Afternoon Snack: Protein bar or mixed nuts and dried fruit",
                    "Dinner: Pasta with ground turkey and salad",
                    "Evening Snack: Cottage cheese with honey and mango"
                ]
            }
        }

        # Select the appropriate diet plan based on the calculated calories
        if calories <= 1500:
            selected_plan = diet_plans["1500"]
        elif calories <= 1800:
            selected_plan = diet_plans["1800"]
        elif calories <= 2000:
            selected_plan = diet_plans["2000"]
        elif calories <= 2200:
            selected_plan = diet_plans["2200"]
        else:
            selected_plan = diet_plans["2500"]

        # Output estimated daily calories, diet plan, and weight guidance
        return {
            "Calculated daily calories": f"{int(calories)} kcal",
            "Adjusted daily calories (base_calories)": f"{int(base_calories)} kcal",
            "Weight Recommendation": weight_goal,
            "Diet Plan": selected_plan,
        }

    except Exception as e:
        return {"response": f"Error parsing input: {str(e)}"}

calorie_cal_tool = Tool(
    name="calorie_calculator_tool",
    func=calorie_calculator_tool,
    description="calculate the calories, also it the respose is in sentence them you should capable to take inputs for form that sentence, then let them know that how much calories they should nee to intake, final create a diet plan for them on the basis of calories they need to intake."
)


tools = [search, retriever_tool, calorie_calculator_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Set up message history for chat
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

class UserInput(BaseModel):
    input_text: str  

@app.post("/generateanswer")
async def generate_answer(user_input: UserInput):
    try:
        response = agent_with_chat_history.invoke(
            {"input": user_input.input_text},
            config={"configurable": {"session_id": "test123"}}
        )

        if response and "output" in response:
            return {"response": response["output"]}
        else:
            return {"response": "No response generated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
