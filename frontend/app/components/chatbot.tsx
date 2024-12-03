// 'use client';
// import React, { useState } from 'react';
// import axios from 'axios';

// const Chatbot = () => {
//   const [messages, setMessages] = useState<{ user: string; bot: string }[]>([]);
//   const [input, setInput] = useState('');
//   const [isOpen, setIsOpen] = useState(false);

//   const handleToggle = () => {
//     setIsOpen(!isOpen);
//   };

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (input.trim()) {
//       setMessages([...messages, { user: input, bot: '...' }]); // Show loading state
//       setInput('');

//       try {
//         const response = await axios.post('http://localhost:8080/chat', {
//           input,
//           // Send any additional parameters for diet calculations if needed
//           age: 25,  // Replace with dynamic values if necessary
//           weight: 70,
//           height: 175,
//           gender: 'male',
//           activity_level: 'moderately active',
//         });

//         // Update the bot's message with the response
//         setMessages((prev) => [...prev, { user: input, bot: response.data.bot_response}]);
//       } catch (error) {
//         console.error(error);
//         setMessages((prev) => [...prev, { user: input, bot: 'Error processing your request.' }]);
//       }
//     }
//   };

//   return (
//     <div className="flex flex-col items-center">
//       {!isOpen && (
//         <button
//           onClick={handleToggle}
//           className="bg-blue-500 hover:bg-blue-600 text-white rounded-full px-6 py-2 mb-4 transition-all duration-300 shadow-lg"
//         >
//           Open Chatbot
//         </button>
//       )}

//       {isOpen && (
//         <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-96 h-96 bg-white rounded-lg shadow-lg p-4 flex flex-col">
//           <h2 className="text-2xl font-bold text-center text-gray-800">Chatbot</h2>
//           <div className="h-3/4 overflow-y-auto my-2 bg-gray-100 rounded-lg p-2 shadow-inner">
//             {messages.map((msg, index) => (
//               <div key={index} className="mb-2">
//                 <div className="text-blue-600 font-semibold">User: {msg.user}</div>
//                 <div className="text-green-600">Bot: {msg.bot}</div>
//               </div>
//             ))}
//           </div>
//           <form onSubmit={handleSubmit} className="flex mt-2">
//             <input
//               type="text"
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               className="flex-1 border border-gray-300 rounded-lg p-2 bg-gray-200 text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-400"
//               placeholder="Type a message..."
//             />
//             <button
//               type="submit"
//               className="ml-2 bg-blue-500 hover:bg-blue-600 rounded-lg px-4 py-2 transition-all duration-300"
//             >
//               Send
//             </button>
//           </form>
//           <button
//             onClick={handleToggle}
//             className="mt-2 bg-red-500 hover:bg-red-600 text-white rounded-full px-4 py-1 transition-all duration-300"
//           >
//             Close Chatbot
//           </button>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Chatbot;