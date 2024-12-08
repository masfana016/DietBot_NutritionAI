[tool.poetry]
name = "calorie-advisor"
version = "0.1.0"
description = ""
authors = ["masfana016 <masfanasrullahansari123@gmail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
langchain = "^0.3.4"
langchain-openai = "^0.2.3"
langchain-core = "^0.3.13"
langchain-community = "^0.3.3"
langchain-google-genai = "^2.0.1"
python-dotenv = "^1.0.1"
beautifulsoup4 = "^4.12.3"
faiss-cpu = "^1.9.0"
uvicorn = {version = "^0.32.0", extras = ["standard"]}
fastapi = "^0.115.3"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"


python -m uvicorn <app_module_name>:<app_instance_name> --port 8080 --reload



*FrontEnd*

'use client';
import React, { useState } from 'react';
import axios from 'axios';

const Chatbot = () => {
  const [messages, setMessages] = useState<{ user: string; bot: string }[]>([]);
  const [input, setInput] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      setMessages([...messages, { user: input, bot: '...' }]); // Show loading state
      setInput('');

      try {
        const response = await axios.post('http://localhost:8003/chat', {
          input,
          // Send any additional parameters for diet calculations if needed
          age: 25,  // Replace with dynamic values if necessary
          weight: 70,
          height: 175,
          gender: 'male',
          activity_level: 'moderately active',
        });

        // Update the bot's message with the response
        setMessages((prev) => [...prev, { user: input, bot: response.data.bot_response}]);
      } catch (error) {
        console.error(error);
        setMessages((prev) => [...prev, { user: input, bot: 'Error processing your request.' }]);
      }
    }
  };

  return (
    <div className="flex flex-col items-center">
      {!isOpen && (
        <button
          onClick={handleToggle}
          className="bg-blue-500 hover:bg-blue-600 text-white rounded-full px-6 py-2 mb-4 transition-all duration-300 shadow-lg"
        >
          Open Chatbot
        </button>
      )}

      {isOpen && (
        <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-96 h-96 bg-white rounded-lg shadow-lg p-4 flex flex-col">
          <h2 className="text-2xl font-bold text-center text-gray-800">Chatbot</h2>
          <div className="h-3/4 overflow-y-auto my-2 bg-gray-100 rounded-lg p-2 shadow-inner">
            {messages.map((msg, index) => (
              <div key={index} className="mb-2">
                <div className="text-blue-600 font-semibold">User: {msg.user}</div>
                <div className="text-green-600">Bot: {msg.bot}</div>
              </div>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="flex mt-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 border border-gray-300 rounded-lg p-2 bg-gray-200 text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-400"
              placeholder="Type a message..."
            />
            <button
              type="submit"
              className="ml-2 bg-blue-500 hover:bg-blue-600 rounded-lg px-4 py-2 transition-all duration-300"
            >
              Send
            </button>
          </form>
          <button
            onClick={handleToggle}
            className="mt-2 bg-red-500 hover:bg-red-600 text-white rounded-full px-4 py-1 transition-all duration-300"
          >
            Close Chatbot
          </button>
        </div>
      )}
    </div>
  );
};

export default Chatbot;








diet_plan = {}

if "gain" in weight_goal.lower():
    # Weight gain plan
    diet_plan = {
        "Breakfast": {
            "Calories": int(base_calories * 0.25),
            "Suggestions": [
                "Peanut butter banana smoothie with oats",
                "Scrambled eggs with cheese and whole grain toast",
                "Chickpea pancakes topped with avocado"
            ]
        },
        "Lunch": {
            "Calories": int(base_calories * 0.35),
            "Suggestions": [
                "Grilled chicken breast with brown rice and mixed vegetables",
                "Quinoa salad with black beans, corn, and avocado",
                "Beef stir-fry with vegetables and whole grain noodles"
            ]
        },
        "Dinner": {
            "Calories": int(base_calories * 0.30),
            "Suggestions": [
                "Baked salmon with quinoa and asparagus",
                "Lentil curry with brown rice",
                "Stuffed bell peppers with ground turkey and cheese"
            ]
        },
        "Snacks": {
            "Calories": int(base_calories * 0.10),
            "Suggestions": [
                "Protein bar or energy balls",
                "Dried fruits and nuts mix",
                "Greek yogurt with honey and granola"
            ]
        }
    }
elif "maintain" in weight_goal.lower():
    # Weight maintenance plan
    diet_plan = {
        "Breakfast": {
            "Calories": int(base_calories * 0.20),
            "Suggestions": [
                "Greek yogurt with berries and nuts",
                "Whole-grain toast with almond butter and a banana",
                "Avocado toast with a poached egg"
            ]
        },
        "Lunch": {
            "Calories": int(base_calories * 0.30),
            "Suggestions": [
                "Chicken wrap with mixed veggies and hummus",
                "Vegetable and chickpea salad with feta",
                "Tofu stir-fry with brown rice and vegetables"
            ]
        },
        "Dinner": {
            "Calories": int(base_calories * 0.30),
            "Suggestions": [
                "Grilled salmon with roasted sweet potatoes and green beans",
                "Turkey burger on a whole grain bun with a side salad",
                "Vegetable quinoa bowl with tahini dressing"
            ]
        },
        "Snacks": {
            "Calories": int(base_calories * 0.20),
            "Suggestions": [
                "Hummus with carrot and cucumber sticks",
                "Apple slices with peanut butter",
                "Handful of mixed nuts"
            ]
        }
    }
elif "loss" in weight_goal.lower():
    # Weight loss plan
    diet_plan = {
        "Breakfast": {
            "Calories": int(base_calories * 0.20),
            "Suggestions": [
                "Smoothie with spinach