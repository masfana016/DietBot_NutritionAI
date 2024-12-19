"use client";

import { useState } from "react";
import Typewriter from "typewriter-effect";

// Define the Message type
interface Message {
  type: "user" | "bot";
  text: string;
}

const ChatPage = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!input.trim()) return;

    // Add user input to chat
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: "user", text: input },
    ]);

    setLoading(true);

    try {
      // Make POST request to the FastAPI endpoint
      const response = await fetch("http://127.0.0.1:8001/generateanswer", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_text: input }),
      });

      const data = await response.json();

      if (data.response) {
        // Add bot response to chat
        setMessages((prevMessages) => [
          ...prevMessages,
          { type: "bot", text: data.response },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { type: "bot", text: "No response generated." },
        ]);
      }
    } catch (err) {
      // Handle errors
      console.log("error", err);
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: "bot", text: "Error: Unable to fetch response." },
      ]);
    }

    setInput(""); // Clear input field
    setLoading(false); // Stop loading
  };

  return (
    <div
      className="flex flex-col items-center justify-center min-h-screen bg-cover bg-center py-8"
      style={{ backgroundImage: "url(/newBg.jpg)" }}
    >
      <div className="m-4 w-full">
        <section className="mb-6  text-center">
          <h1 className="text-3xl font-bold text-white">Nutritionist</h1>
          <h6 className="text-sm text-white">
            Your AI Powered nutrition planner assistant
          </h6>
        </section>

        <section className="mb-4 w-auto mx-4 sm:w-4/6 md:w-3/6 sm:m-auto">
          <div className="bg-white bg-opacity-20 backdrop-blur-md border border-gray-300 rounded-md p-3 h-96 overflow-y-auto flex flex-col">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`my-2 p-2 rounded-lg ${
                  msg.type === "user"
                    ? "bg-red-600 bg-opacity-80 backdrop-blur-md text-white self-end"
                    : "bg-gray-200 text-black self-start"
                }`}
              >
                {msg.type === "user" ? (
                  msg.text
                ) : index === messages.length - 1 ? (
                  <Typewriter
                    options={{
                      strings: [msg.text],
                      autoStart: true,
                      cursor: "",
                      delay: 0,
                      loop: false,
                      deleteSpeed: Infinity,
                    }}
                  />
                ) : (
                  msg.text
                )}
              </div>
            ))}
          </div>
          <section className="flex flex-col w-full mt-2">
            <form onSubmit={handleSubmit} className="flex flex-col">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="p-2 border border-gray-300 rounded-lg mb-2"
                placeholder="What is your goal?"
                required
              />
              <button
                type="submit"
                className={`p-2 rounded-lg flex cursor-pointer items-center justify-center ${
                  loading ? "bg-gray-500" : "bg-red-600 hover:bg-red-700"
                } text-white`}
                disabled={loading}
              >
                {loading ? (
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 0016 0H4z"
                    />
                  </svg>
                ) : (
                  "Submit"
                )}
              </button>
            </form>
          </section>
        </section>
      </div>
    </div>
  );
};

export default ChatPage;



























// 'use client'

// import React, { useState } from 'react'
// import Image from 'next/image';
// import img from "./../public/robot.jpg"
// import img1 from "./../public/human.jpg"


// type ChatMessageType = {
//   role: string;
//   content: string;
// }



// // export async function POST(req: Request): Promise<NextResponse> {
// //   try {
// //     const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
// //     const model = await genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// //     return NextResponse.json({
// //       success: true,
// //       data: model,
// //     });
// //   } catch (error) {
// //     return NextResponse.json({
// //       success: false,
// //       error: error.message,
// //     });
// //   }
// // }


// export default function page() {
//   const [userInput, setUserInput] = useState<string>('')
//   const [chatHistory, setChatHistory] = useState<ChatMessageType[]>([])
//   const [isLoading, setIsLoading] = useState<boolean>(false)



//   const handleUserInput = async () => {
//     if (!userInput) return;

//     setIsLoading(true);

//     // Add the user input to the chat history
//     setChatHistory((prevChat) => [
//       ...prevChat,
//       { role: "user", content: userInput },
//     ]);

//     try {
//       const response = await fetch("http://127.0.0.1:8001/generateanswer", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ input_text: userInput }),
//       });

//       const data = await response.json();

//       if (data.response) {
//         setChatHistory((prevChat) => [
//           ...prevChat,
//           { role: "assistant", content: data.response },
//         ]);
//       } else {
//         setChatHistory((prevChat) => [
//           ...prevChat,
//           { role: "assistant", content: "No response generated." },
//         ]);
//       }
//     } catch (error) {
//       setChatHistory((prevChat) => [
//         ...prevChat,
//         { role: "assistant", content: "Error: Unable to fetch response." },
//       ]);
//     }

//     setUserInput(""); // Clear the input field
//     setIsLoading(false); // Stop loading
//   };

//   return (
//     <div className='bg-gray-100 min-h-screen flex flex-col justify-center items-center'>

//       <div className='w-full max-w-screen-md bg-white rounded-t-xl rounded-b-2xl shadow-xl shadow-sky-800'>
//         <div className='flex items-center justify-between mb-4'>
//           {/* Header */}
//           <div className='w-full p-3 bg-gradient-to-r from-[#7671db] via-[#918de0] to-[#b9b2fb] rounded-t-xl'>
//             {/* <div className='text-2xl font-bold px-1 m-2 text-violet-600 mb-4'>AI Chatbot</div> */}
//             <div>
//               <div className="flex justify-end space-x-6">
//                 <div className="w-4 h-4 bg-violet-50 rounded-full"></div>
//                 <div className="w-4 h-4 bg-violet-50 rounded-full"></div>
//                 <div className="w-4 h-4 bg-[#7671db] rounded-full"></div>
//               </div>
//             </div>
//           </div>
//         </div>
//         {/* Messages */}
//         <div className='mb-4' style={{ height: "400px", overflow: 'auto' }}>
//           {chatHistory.map((message, index) => (
//             <div
//               key={index}
//               className={`flex items-start mb-2 text-sm text-gray-600 ${message.role === 'user' ? 'justify-end' : 'justify-start'
//                 }`}
//             >
//               {/* Image for User or Assistant */}
//               <div className="flex-shrink-0">
//                 {message.role === 'user' ? (
//                   <Image
//                     src= {img1} // Replace with the actual human image path
//                     alt="User"
//                     className="h-8 w-8 rounded-full mx-2"
//                   />
//                 ) : (
//                   <Image
//                     src={img} // Replace with the actual robot image path
//                     alt="Assistant"
//                     className="h-8 w-8 rounded-full mx-2"
//                   />
//                 )}
//               </div>

//               {/* Message Content */}
//               <div
//                 className={`inline-block p-2 rounded-md max-w-md ${message.role === 'user'
//                   ? 'bg-gradient-to-r from-[#7671db] to-[#4f65d2] text-white'
//                   : 'bg-gradient-to-r from-[#b9c3e8] to-[#697bd2] text-gray-800'
//                   }`}
//               >
//                 {message.content}
//               </div>
//             </div>
//           ))}
//         </div>
//         {/* Input Area of Chatbot */}
//         <div className='flex p-4'>
//           <input
//             type='text'
//             value={userInput}
//             onChange={(e) => setUserInput(e.target.value)}
//             className='flex-1 w-full px-3 py-2 text-gray-700 border border-gray-300 rounded-l-lg'
//             placeholder='Ask me anything...'
//           />
//           {isLoading ? (
//             <div className='bg-[#4bb9db] text-white p-2 rounded-sm shadow-l-2xl shadow-[#4bb9db] animate-pulse'>
//               Loading...
//             </div>
//           ) : (
//             <button
//               disabled={isLoading}
//               onClick={handleUserInput}
//               className={`px-4 py-2 font-sans ${isLoading ? 'opacity-50 cursor-not-allowed' : 'bg-[#1466a9] hover:bg-cyan-500 hover:duration-300 text-white p-2 border-transparent rounded-r-lg'}`}
//             >
//               Send
//             </button>
//           )}

//         </div>
//         <div className=' bg-[#99bfda] py-3 rounded-b-2xl shadow-xl shadow-[#a2c0d7]'>

//         </div>
//       </div>
//     </div>
//   )
// }
