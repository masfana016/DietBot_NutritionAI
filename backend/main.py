from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph # type
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from fastapi import FastAPI, HTTPException
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

loader1 = WebBaseLoader("https://www.healthline.com/nutrition/1500-calorie-diet#foods-to-eat")
# loader2 = WebBaseLoader("https://www.msdmanuals.com/home")
# loader3 = WebBaseLoader("https://www.eatingwell.com/category/4305/weight-loss-meal-plans/")
docs1 = loader1.load()
# docs2 = loader2.load()
# docs3 = loader3.load()
# combined_docs = docs1 + docs2 + docs3
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs1)
vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "healthline_search",
    "Search for information about healthline, food, diet and nutrition. For any questions about food and nutrition, healthy diet related and just answer the question, don't explain much, you must use this tool!",
)

def calorie_calculator_tool(gender: str, weight: float, height: float, age: int, activity_level: str) -> dict:
    """
    Tool to compute daily caloric needs based on user input using the Harris-Benedict Equation.
    
    Args:
        sex (str): The user's gender ('male' or 'female').
        weight (float): The user's weight in kilograms.
        height (float): The user's height in centimeters.
        age (int): The user's age in years.
        activity_level (str): The user's activity level ('sedentary', 'light', 'moderate', 'active', 'very_active').
    
    Returns:
        dict: A dictionary containing:
            - 'bmr': Basal Metabolic Rate (calories burned at rest).
            - 'daily_calories': Estimated daily caloric needs based on activity level.
            - 'maintenance_plan': A guide for maintaining current weight.
    """
    def calculate_daily_calories(gender, weight, height, age, activity_level):
        # Basal Metabolic Rate (BMR) calculation
        if gender.lower() == "male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        elif gender.lower() == "female":
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            raise ValueError("Invalid sex. Please use 'male' or 'female'.")

        # Activity multipliers
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        
        if activity_level not in activity_multipliers:
            raise ValueError("Invalid activity level. Choose from 'sedentary', 'light', 'moderate', 'active', or 'very_active'.")

        # Calculate daily caloric needs
        daily_calories = bmr * activity_multipliers[activity_level]

        base_calories = daily_calories  # Initialize with the calculated calories

        if gender == "female":
            if activity_level == "lightly":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, daily_calories), 1400)
                elif 7 <= age <= 18:
                    base_calories = min(max(1200, daily_calories), 1800)
                elif 19 <= age <= 60:
                    base_calories = min(max(1600, daily_calories), 2000)
                elif age >= 61:
                    base_calories = max(1600, daily_calories)
            elif activity_level == "active":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, daily_calories), 1600)
                elif 7 <= age <= 18:
                    base_calories = min(max(1600, daily_calories), 2400)
                elif 19 <= age <= 60:
                    base_calories = min(max(1800, daily_calories), 2400)
                elif age >= 61:
                    base_calories = min(max(1800, daily_calories), 2000)

        elif gender == "male":
            if activity_level == "lightly":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, daily_calories), 1400)
                elif 7 <= age <= 18:
                    base_calories = min(max(1400, daily_calories), 2400)
                elif 19 <= age <= 60:
                    base_calories = min(max(2200, daily_calories), 2600)
                elif age >= 61:
                    base_calories = max(2000, daily_calories)
            elif activity_level == "active":
                if 2 <= age <= 6:
                    base_calories = min(max(1000, daily_calories), 1800)
                elif 7 <= age <= 18:
                    base_calories = min(max(1600, daily_calories), 3200)
                elif 19 <= age <= 60:
                    base_calories = min(max(2400, daily_calories), 3000)
                elif age >= 61:
                    base_calories = min(max(2200, daily_calories), 2600)

        # Determine if the user needs to gain, lose, or maintain weight
        if daily_calories < base_calories:
            weight_goal = "You should increase your calorie intake to gain weight."
        elif daily_calories > base_calories:
            weight_goal = "You should decrease your calorie intake to lose weight."
        else:
            weight_goal = "Your calorie intake is appropriate for maintaining your current weight."

        # Define diet plans
        diet_plans = {
            "1500": {
                "goal": "Weight Loss",
                "plan": [
                    "Breakfast: Greek yogurt with berries and chia seeds or oatmeal with sliced banana",
                    "Morning Snack: Apple with almond butter or a handful of mixed nuts",
                    "Lunch: Grilled chicken salad with greens and vinaigrette or quinoa salad with chickpeas and cucumber",
                    "Afternoon Snack: Cottage cheese with cucumber or carrot sticks with hummus",
                    "Dinner: Baked salmon, asparagus, and quinoa or stir-fried tofu with broccoli and brown rice",
                    "Evening Snack: Almonds or a small piece of dark chocolate"
                ]
            },
            "1800": {
                "goal": "Weight Maintenance",
                "plan": [
                    "Breakfast: Scrambled eggs with spinach, whole-grain toast or smoothie with spinach and protein powder",
                    "Morning Snack: Orange and walnuts or Greek yogurt with honey",
                    "Lunch: Turkey wrap with hummus and veggies or lentil soup with whole-grain bread",
                    "Afternoon Snack: Banana with peanut butter or rice cakes with avocado",
                    "Dinner: Grilled chicken, sweet potato, and broccoli or baked tilapia with quinoa and green beans",
                    "Evening Snack: Cottage cheese with berries or air-popped popcorn"
                ]
            },
            "2000": {
                "goal": "Moderate Weight Gain",
                "plan": [
                    "Breakfast: Overnight oats with banana and peanut butter or avocado toast with eggs",
                    "Morning Snack: Smoothie with protein powder or a protein bar",
                    "Lunch: Brown rice bowl with black beans and salsa or chicken stir-fry with vegetables",
                    "Afternoon Snack: Toast with cottage cheese and tomatoes or fruit salad",
                    "Dinner: Steak, mashed potatoes, and green beans or chicken curry with brown rice",
                    "Evening Snack: Greek yogurt with honey and pumpkin seeds or protein shake"
                ]
            },
            "2200": {
                "goal": "Active Weight Maintenance/Gain",
                "plan": [
                    "Breakfast: Omelet with veggies and whole-grain toast or smoothie bowl with fruits and granola",
                    "Morning Snack: Greek yogurt with granola and blueberries or nut butter on whole-grain bread",
                    "Lunch: Tuna wrap with veggies or quinoa salad with chickpeas and feta",
                    "Afternoon Snack: Apple and almonds or veggie sticks with hummus",
                    "Dinner: Roasted chicken, brown rice, and carrots or fish tacos with cabbage slaw",
                    "Evening Snack: Dark chocolate with walnuts or a handful of dried fruit"
                ]
            },
            "2500": {
                "goal": "High-Calorie for Weight Gain",
                "plan": [
                    "Breakfast: Smoothie bowl with peanut butter and granola or pancakes with maple syrup",
                    "Morning Snack: Crackers with cheese and apple or energy bites with oats and honey",
                    "Lunch: Quinoa bowl with chickpeas and roasted veggies or burrito with beans and cheese",
                    "Afternoon Snack: Protein bar or mixed nuts and dried fruit or yogurt with granola",
                    "Dinner: Pasta with ground turkey and salad or lamb kebabs with rice and grilled vegetables",
                    "Evening Snack: Cottage cheese with honey and mango or fruit and nut mix"
                ]
            }
        }

        # Select the appropriate diet plan based on the calculated calories
        if daily_calories <= 1500:
            selected_diet_plan = diet_plans["1500"]
        elif daily_calories <= 1800:
            selected_diet_plan = diet_plans["1800"]
        elif daily_calories <= 2000:
            selected_diet_plan = diet_plans["2000"]
        elif daily_calories <= 2200:
            selected_diet_plan = diet_plans["2200"]
        else:
            selected_diet_plan = diet_plans["2500"]
        
        # Create a maintenance plan
        maintenance_plan = {
            "maintain": round(daily_calories),
            "gain_weight": round(daily_calories + 500),
            "lose_weight": round(daily_calories - 500)
        }
        
        return {
            "bmr": round(bmr, 2),
            "daily_calories": round(daily_calories, 2),
            "maintenance_plan": maintenance_plan,
            "weight_goal": weight_goal,
            "selected_diet_plan": selected_diet_plan
        }

    # Return calculated daily calories
    return calculate_daily_calories(gender, weight, height, age, activity_level)

# def weight_goal_and_diet_plan(daily_calories, gender, age, activity_level):
    
#     """
#     This function takes the gender, age, activity level, anddaily_caloriesas input,
#     and returns a weight goal along with a suggested diet plan based on the calorie intake.

#     :param gender: str - "female" or "male"
#     :param age: int - the age of the user
#     :param activity_level: str - "lightly" or "active"
#     :param calories: int - the calculated calorie intake
#     :return: dict - containing weight goal and suggested diet plan
#     """

#     base_calories = daily_calories  # Initialize with the calculated calories

#     if gender == "female":
#         if activity_level == "lightly":
#             if 2 <= age <= 6:
#                 base_calories = min(max(1000, daily_calories), 1400)
#             elif 7 <= age <= 18:
#                 base_calories = min(max(1200, daily_calories), 1800)
#             elif 19 <= age <= 60:
#                 base_calories = min(max(1600, daily_calories), 2000)
#             elif age >= 61:
#                 base_calories = max(1600, daily_calories)
#         elif activity_level == "active":
#             if 2 <= age <= 6:
#                 base_calories = min(max(1000, daily_calories), 1600)
#             elif 7 <= age <= 18:
#                 base_calories = min(max(1600, daily_calories), 2400)
#             elif 19 <= age <= 60:
#                 base_calories = min(max(1800, daily_calories), 2400)
#             elif age >= 61:
#                 base_calories = min(max(1800, daily_calories), 2000)

#     elif gender == "male":
#         if activity_level == "lightly":
#             if 2 <= age <= 6:
#                 base_calories = min(max(1000, daily_calories), 1400)
#             elif 7 <= age <= 18:
#                 base_calories = min(max(1400, daily_calories), 2400)
#             elif 19 <= age <= 60:
#                 base_calories = min(max(2200, daily_calories), 2600)
#             elif age >= 61:
#                 base_calories = max(2000, daily_calories)
#         elif activity_level == "active":
#             if 2 <= age <= 6:
#                 base_calories = min(max(1000, daily_calories), 1800)
#             elif 7 <= age <= 18:
#                 base_calories = min(max(1600, daily_calories), 3200)
#             elif 19 <= age <= 60:
#                 base_calories = min(max(2400, daily_calories), 3000)
#             elif age >= 61:
#                 base_calories = min(max(2200, daily_calories), 2600)

#     # Determine if the user needs to gain, lose, or maintain weight
#     if daily_calories < base_calories:
#         weight_goal = "You should increase your calorie intake to gain weight."
#     elif daily_calories > base_calories:
#         weight_goal = "You should decrease your calorie intake to lose weight."
#     else:
#         weight_goal = "Your calorie intake is appropriate for maintaining your current weight."

#     # Define diet plans
#     diet_plans = {
#         "1500": {
#             "goal": "Weight Loss",
#             "plan": [
#                 "Breakfast: Greek yogurt with berries and chia seeds or oatmeal with sliced banana",
#                 "Morning Snack: Apple with almond butter or a handful of mixed nuts",
#                 "Lunch: Grilled chicken salad with greens and vinaigrette or quinoa salad with chickpeas and cucumber",
#                 "Afternoon Snack: Cottage cheese with cucumber or carrot sticks with hummus",
#                 "Dinner: Baked salmon, asparagus, and quinoa or stir-fried tofu with broccoli and brown rice",
#                 "Evening Snack: Almonds or a small piece of dark chocolate"
#             ]
#         },
#         "1800": {
#             "goal": "Weight Maintenance",
#             "plan": [
#                 "Breakfast: Scrambled eggs with spinach, whole-grain toast or smoothie with spinach and protein powder",
#                 "Morning Snack: Orange and walnuts or Greek yogurt with honey",
#                 "Lunch: Turkey wrap with hummus and veggies or lentil soup with whole-grain bread",
#                 "Afternoon Snack: Banana with peanut butter or rice cakes with avocado",
#                 "Dinner: Grilled chicken, sweet potato, and broccoli or baked tilapia with quinoa and green beans",
#                 "Evening Snack: Cottage cheese with berries or air-popped popcorn"
#             ]
#         },
#         "2000": {
#             "goal": "Moderate Weight Gain",
#             "plan": [
#                 "Breakfast: Overnight oats with banana and peanut butter or avocado toast with eggs",
#                 "Morning Snack: Smoothie with protein powder or a protein bar",
#                 "Lunch: Brown rice bowl with black beans and salsa or chicken stir-fry with vegetables",
#                 "Afternoon Snack: Toast with cottage cheese and tomatoes or fruit salad",
#                 "Dinner: Steak, mashed potatoes, and green beans or chicken curry with brown rice",
#                 "Evening Snack: Greek yogurt with honey and pumpkin seeds or protein shake"
#             ]
#         },
#         "2200": {
#             "goal": "Active Weight Maintenance/Gain",
#             "plan": [
#                 "Breakfast: Omelet with veggies and whole-grain toast or smoothie bowl with fruits and granola",
#                 "Morning Snack: Greek yogurt with granola and blueberries or nut butter on whole-grain bread",
#                 "Lunch: Tuna wrap with veggies or quinoa salad with chickpeas and feta",
#                 "Afternoon Snack: Apple and almonds or veggie sticks with hummus",
#                 "Dinner: Roasted chicken, brown rice, and carrots or fish tacos with cabbage slaw",
#                 "Evening Snack: Dark chocolate with walnuts or a handful of dried fruit"
#             ]
#         },
#         "2500": {
#             "goal": "High-Calorie for Weight Gain",
#             "plan": [
#                 "Breakfast: Smoothie bowl with peanut butter and granola or pancakes with maple syrup",
#                 "Morning Snack: Crackers with cheese and apple or energy bites with oats and honey",
#                 "Lunch: Quinoa bowl with chickpeas and roasted veggies or burrito with beans and cheese",
#                 "Afternoon Snack: Protein bar or mixed nuts and dried fruit or yogurt with granola",
#                 "Dinner: Pasta with ground turkey and salad or lamb kebabs with rice and grilled vegetables",
#                 "Evening Snack: Cottage cheese with honey and mango or fruit and nut mix"
#             ]
#         }
#     }

#     # Select the appropriate diet plan based on the calculated calories
#     if daily_calories <= 1500:
#         selected_diet_plan = diet_plans["1500"]
#     elif daily_calories <= 1800:
#         selected_diet_plan = diet_plans["1800"]
#     elif daily_calories <= 2000:
#         selected_diet_plan = diet_plans["2000"]
#     elif daily_calories <= 2200:
#         selected_diet_plan = diet_plans["2200"]
#     else:
#         selected_diet_plan = diet_plans["2500"]

#     return {
#         "weight_goal": weight_goal,
#         "selected_diet_plan": selected_diet_plan
#     }


tools = [search, retriever_tool, calorie_calculator_tool]


llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content='''You are a helpful customer support assistant for human calorie calculation.
            You need to gather the following information from the user:
            - Person's age, weight, height, gender /pronouns (e.g., she, he, or similar), and activity level (e.g., sedentary, moderate, active, very active).
            
            Based on their gender/pronouns, infer if the user is male or female. Do this implicitly and avoid explicitly asking about gender. 
            Similarly, if they provide information about their daily routine or habits, interpret their activity level. 
            
            If you are unable to discern any of this information, politely ask them to clarify! 
            Never make random guesses if the details remain unclear.

            Once all the necessary information is gathered, call the relevant tool to perform the calorie calculation.

            **Tool to check If user need to gain weight or loss weight**
             - **adjust_calories_for_goal**: Adjust the daily caloric needs based on the user's goal (gain or lose weight). Use the user's provided age, gender, activity level, and calories to calculate adjusted caloric intake for the goal.

            **Important Tools for Diet and Health Information**:
            - **TavilySearchResults**: Use this tool to search for health, food, diet, and nutrition information by making API calls with `TAVILY_API_KEY`. This will help you gather relevant resources when a user asks for diet suggestions or general nutrition-related queries.
            
            - **Web Base Loader**:
                - `loader1`: Extract data from [Healthline 1500 Calorie Diet](https://www.healthline.com/nutrition/1500-calorie-diet#foods-to-eat).
                
            - **Document Handling**:
                - Use `WebBaseLoader` to load content from the above health-related sites.
                - Combine the documents from all three sources using `docs1 for a broader perspective.
                - Split the doc1 into smaller chunks with `RecursiveCharacterTextSplitter` to ensure the content is manageable and precise.
                - Use `FAISS` for vectorizing documents and creating a retriever tool, which can search for the most relevant information.

            **calories calculated tool**:
            - **Calorie Calculation**: Once you have the user's details, use the following logic to calculate the required daily calorie intake based on gender, age, weight, and activity level.
            - **Weight Goal and Diet Plan Tool**: After calculating the calories, use this tool to determine if the user needs to gain, lose, or maintain weight, and provide them with a personalized diet plan based on their calorie needs.
                                 
            **Retriever Tool for Searching Information**:
            - Create a `retriever_tool` from the vector retriever to search through the documents. For any user questions related to food, nutrition, health or healthy diets, use the retriever to fetch relevant content from Healthline, MSD Manual, or EatingWell.
            
            **When answering questions**:
            - Always use the retriever tool to provide concise and relevant answers about food, nutrition, and diet. Don't over-explain; just provide the information needed. 
            After gathering the user's details and answering any inquiries, proceed to calculate the user's calorie needs and provide a personalized diet plan.
''')


# Node
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder: StateGraph = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory: MemorySaver = MemorySaver()
react_graph_memory: CompiledStateGraph = builder.compile(checkpointer=memory)

class UserInput(BaseModel):
    input_text: str 

# API endpoint
@app.post("/generateanswer")
async def generate_answer(user_input: UserInput):
    try:
        messages = [HumanMessage(content=user_input.input_text)]
        response = react_graph_memory.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})

        # Extract the response from the graph output
        if response and "messages" in response:
            # Extract the last message (assistant's response)
            assistant_response = response["messages"][-1].content
            return {"response": assistant_response}
        else:
            return {"response": "No response generated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)



# # Specify a thread
# config1 = {"configurable": {"thread_id": "1"}}


# messages = [HumanMessage(content="How much would I save by switching to solar panels if my monthly electricity cost is $200?")]
# messages = react_graph_memory.invoke({"messages": messages}, config1)
# for m in messages['messages']:
#     m.pretty_print()