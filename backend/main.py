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
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from fastapi import FastAPI, HTTPException
import uvicorn

load_dotenv()

app = FastAPI()

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
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        elif gender.lower() == "female":
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
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
        
        # Create a maintenance plan
        maintenance_plan = {
            "maintain": round(daily_calories),
            "gain_weight": round(daily_calories + 500),
            "lose_weight": round(daily_calories - 500)
        }
        
        return {
            "bmr": round(bmr, 2),
            "daily_calories": round(daily_calories, 2),
            "maintenance_plan": maintenance_plan
        }

    # Return calculated daily calories
    return calculate_daily_calories(gender, weight, height, age, activity_level)


tools = [search, retriever_tool, calorie_calculator_tool]

llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content='''You are a helpful customer support assistant for human calorie calculation.
            You need to gather the following information from the user:
            - Person's age, weight, height, pronouns (e.g., she, he, or similar), and activity level (e.g., sedentary, moderate, active, very active).
            
            Based on their pronouns, infer if the user is male or female. Do this implicitly and avoid explicitly asking about gender. 
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
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)



# # Specify a thread
# config1 = {"configurable": {"thread_id": "1"}}


# messages = [HumanMessage(content="How much would I save by switching to solar panels if my monthly electricity cost is $200?")]
# messages = react_graph_memory.invoke({"messages": messages}, config1)
# for m in messages['messages']:
#     m.pretty_print()
