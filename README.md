RUBY is an advanced AI-powered assistant designed to enhance virtual meetings through real-time Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) capabilities. The project aims to solve a common productivity challenge—manually managing and reviewing large volumes of meeting data—by offering a smarter, more intuitive alternative.

🔍 Key Features:
Live Transcription: RUBY accurately converts spoken content into structured, readable text using speech recognition techniques.


Automated Summarization: It condenses long meetings into concise summaries, highlighting key points and takeaways.


Semantic Analysis: The assistant understands and categorizes content (e.g., opening remarks, conclusions, who said what), helping users grasp context at a glance.


Query Resolution: Users can ask questions like “What did John say about the budget?” and receive accurate, context-aware answers.


Interactive Chat Interface: Built using Streamlit, the frontend offers a clean, chat-based UI for real-time interaction.



🧠 Tech Stack & Core Logic:
Backend: Python, LangChain, Chroma DB (for vector storage), and Ollama (LLM backend).


Frontend: Streamlit with custom CSS for UI enhancements.


Document Handling: Meeting transcripts are loaded as PDFs, split into chunks, and embedded using SentenceTransformer.


Retrieval-Augmented Generation (RAG): This model retrieves the most relevant segments from past transcripts and uses a generative model to formulate accurate, context-sensitive responses.
