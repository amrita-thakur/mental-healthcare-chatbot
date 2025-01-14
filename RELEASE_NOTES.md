## [v0.3.0]

### Key Changes:
- **Optimize Response Time for Chatbot by Using RAPTOR for Retrieval:**
  - Replaced the previous retrieval system with RAPTOR for faster response times without compromising accuracy.
  - Improved chatbot performance and reduced response time.

- **Optimize Chatbot Embedding Generation:**
  - Precomputed embeddings are now saved and loaded from a pickle file during initialization, optimizing the performance and reducing computational load.


## [v0.2.0]

### Key Changes:
- **Data Integration with NIMH Datasets:**
  - Integrated curated datasets from the National Institute of Mental Health (NIMH) for building the FAISS vector store.
  - Improved data storage and retrieval performance with the new `faiss_vectorstore.pkl` file format.


## [v0.1.0]

### Key Features:
- **FAISS Vector Store:**
  - Implemented FAISS for efficient retrieval based on the **[Defining Mental Health and Mental Illness](https://www.researchgate.net/publication/328248529_Defining_mental_health_and_mental_illness)** PDF.
  - Enabled chatbot to respond to mental health-related queries.
