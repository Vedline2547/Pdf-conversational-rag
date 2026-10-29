
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
# Initialize the Ollama model
llm = ChatOllama(model="gemma3:1b",temperature=0)
# Test the model
question = "When did India gain independence?"
response = llm.invoke(question)
print("General answer:", response.content)
pdf_path = Path(r"C:\Users\HP\Desktop\New ML\Ollama\RAGPaper+(1).pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")
pdf_reader = PyPDFLoader(str(pdf_path))
documents = pdf_reader.load()
print("PDF pages loaded:", len(documents))
# Split the PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("Text chunks created:", len(chunks))
#  Create Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create the FAISS vector database
db = FAISS.from_documents(documents=chunks,embedding=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
# Condense follow-up questions
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question,
rephrase the follow-up question into a standalone question.
Chat History:
{chat_history}
Follow-up Input:
{question}
Standalone Question:""")
# Build the conversational RAG chain
qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True,verbose=False)
# Ask a question about the PDF
chat_history = []
query = "What are the acknowledgments in this paper?"
result = qa.invoke({"question": query,"chat_history": chat_history})
print("\nQuestion:", query)
print("\nAnswer:", result["answer"])


