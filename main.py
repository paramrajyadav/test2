from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import OpenAI
from langchain.document_loaders import TextLoader

# Load your documents
loader = TextLoader("path/to/your/documents.txt")
documents = loader.load()

# Generate embeddings for the documents
embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")
vector_store = FAISS.from_documents(documents, embeddings)

# Create a retriever
retriever = vector_store.as_retriever()

# Initialize the OpenAI language model
llm = OpenAI(openai_api_key="your_openai_api_key", model="gpt-4")

# Build the RAG chain using the retriever and language model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 'stuff' means the text retrieved is used as context in the prompt.
    retriever=retriever,
    return_source_documents=True  # Option to see the sources returned
)

# Run a query through the RAG chain
query = "What are the key benefits of using RAG models in NLP?"
response = qa_chain.run(query)

# Print the response
print("Answer:", response['result'])
print("Source documents:", response['source_documents'])
