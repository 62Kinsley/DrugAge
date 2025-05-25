# DrugAge

DrugAge Chatbot Project: Technical Architecture Concept
1. Summary:
This document outlines the technical architecture for an intelligent chatbot interfacing with the DrugAge database. The solution employs a "GPT as coordinator" design pattern, allowing researchers to explore longevity-enhancing compounds through natural language queries with accurate data retrieval.
This project aims to build an intelligent chatbot based on the DrugAge dataset to help researchers and life science professionals quickly query and compare the lifespan-extending effects of drugs using natural language. The architecture is designed to be lightweight yet powerful: a large language model (GPT) acts as the coordinator, calling backend tool functions to process the DrugAge data and return evidence-based responses.

2. System Architecture Concept
Core Model: GPT Coordinator + Tool Functions
                           
User Query -> GPT Understands Intent -> Calls Tool Function -> Returns Structured Result -> GPT Generates Response


Advantages of this design:
- Modular Separation: LLM handles understanding and answering, functions handle data - Easy to Extend: Add functionality by expanding functions 
- Natural Language Interface: Users can interact through conversation rather than learning specific query syntax. 
- Accurate Data Handling: Critical data operations performed by purpose-built functions ensure precision.

3. Technology Stack
1)	Frontend Technologies
- Streamlit: Used to build interactive web apps, suitable for lightweight AI tool interfaces. 
- Plotly / Seaborn / Matplotlib: Generate drug effect visualizations and comparison charts.
2)	Backend Technologies
- Python 3.9+: Main language for development and AI interaction. 
- Pandas / NumPy: Load, filter, and analyze the DrugAge CSV dataset. 
- Agno: Lightweight AI framework, connects GPT calls with tool functions. 
- OpenAI API (GPT-3.5 / GPT-4): Core engine for natural language understanding and response generation
3)	Data and Storage
- CSV + Pandas DataFrame: Core structure for loading and querying DrugAge data. 
- SQLite (Optional): For more advanced filtering or persistent storage, if needed in the future.
