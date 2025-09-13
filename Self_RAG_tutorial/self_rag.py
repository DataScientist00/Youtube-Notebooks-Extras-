import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langgraph.graph import END, StateGraph, START

from prompts import DOCUMENT_GRADER_PROMPT, HALLUCINATION_GRADER_PROMPT, ANSWER_GRADER_PROMPT


# Load environment variables (Groq API key, etc.)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Knowledge base sources
SOURCE_LINKS = [
    "https://www.geeksforgeeks.org/dsa/disjoint-set-data-structures/",
    "https://en.wikipedia.org/wiki/Data_science",
]


# Shared state for our workflow
class WorkflowState(TypedDict):
    user_question: str
    answer_draft: str
    retrieved_docs: List[str]
    llm_model: ChatGroq
    retriever: Chroma
    has_hallucination: bool
    is_valid_answer: bool


# Models for grading
class DocRelevanceScore(BaseModel):
    binary_score: str = Field(description="'yes' if document is relevant, otherwise 'no'")


class HallucinationScore(BaseModel):
    binary_score: str = Field(description="'yes' if grounded in facts, otherwise 'no'")


class AnswerValidityScore(BaseModel):
    binary_score: str = Field(description="'yes' if answer addresses the question, otherwise 'no'")


# 1. Load Groq model
def init_groq_model(state: WorkflowState) -> WorkflowState:
    print("---LOADING GROQ MODEL---")
    state["llm_model"] = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    return state


# 2. Create vector database
def prepare_vector_database(state: WorkflowState) -> WorkflowState:
    print("---CREATING VECTOR DATABASE---")
    all_docs = [WebBaseLoader(url).load() for url in SOURCE_LINKS]
    flat_docs = [doc for group in all_docs for doc in group]

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    split_docs = splitter.split_documents(flat_docs)

    vector_db = Chroma.from_documents(
        documents=split_docs,
        collection_name="custom_rag_store",
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    )
    state["retriever"] = vector_db.as_retriever()
    return state


# 3. Retrieve documents
def fetch_relevant_docs(state: WorkflowState) -> WorkflowState:
    print("---RETRIEVING RELEVANT DOCUMENTS---")
    state["retrieved_docs"] = state["retriever"].invoke(state["user_question"])
    return state


# 4. Filter out irrelevant documents
def filter_docs_by_relevance(state: WorkflowState) -> WorkflowState:
    print("---FILTERING DOCUMENTS FOR RELEVANCE---")
    grader = state["llm_model"].with_structured_output(DocRelevanceScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_GRADER_PROMPT),
        ("human", "Document: {document}\n\nQuestion: {question}")
    ])
    evaluation_chain = prompt | grader

    filtered = []
    for doc in state["retrieved_docs"]:
        score = evaluation_chain.invoke({"document": doc.page_content, "question": state["user_question"]})
        if score.binary_score.lower() == "yes":
            filtered.append(doc)

    state["retrieved_docs"] = filtered
    return state


# 5. Decide whether to answer or stop
def should_generate_answer(state: WorkflowState) -> str:
    print("---DECIDING NEXT STEP---")
    return "answer" if state["retrieved_docs"] else "stop"


# 6. Generate answer
def produce_answer(state: WorkflowState) -> WorkflowState:
    print("---GENERATING ANSWER---")
    prompt_template = hub.pull("rlm/rag-prompt")
    rag_chain = prompt_template | state["llm_model"] | StrOutputParser()
    state["answer_draft"] = rag_chain.invoke({
        "context": state["retrieved_docs"],
        "question": state["user_question"]
    })
    return state


# 7. Check for hallucinations
def detect_hallucination(state: WorkflowState) -> WorkflowState:
    print("---CHECKING FOR HALLUCINATIONS---")
    grader = state["llm_model"].with_structured_output(HallucinationScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_GRADER_PROMPT),
        ("human", "Facts: {documents}\n\nGenerated Answer: {generation}")
    ])
    chain = prompt | grader
    result = chain.invoke({
        "documents": state["retrieved_docs"],
        "generation": state["answer_draft"]
    })
    state["has_hallucination"] = (result.binary_score.lower() != "yes")
    return state


# 8. Grade final answer
def validate_answer(state: WorkflowState) -> WorkflowState:
    print("---VALIDATING ANSWER---")
    grader = state["llm_model"].with_structured_output(AnswerValidityScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_GRADER_PROMPT),
        ("human", "Question: {question}\n\nAnswer: {generation}")
    ])
    chain = prompt | grader
    result = chain.invoke({
        "question": state["user_question"],
        "generation": state["answer_draft"]
    })
    state["is_valid_answer"] = (result.binary_score.lower() == "yes")
    return state


# 9. Build graph
def create_workflow():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("init_groq_model", init_groq_model)
    workflow.add_node("prepare_vector_database", prepare_vector_database)
    workflow.add_node("fetch_relevant_docs", fetch_relevant_docs)
    workflow.add_node("filter_docs_by_relevance", filter_docs_by_relevance)
    workflow.add_node("produce_answer", produce_answer)
    workflow.add_node("detect_hallucination", detect_hallucination)
    workflow.add_node("validate_answer", validate_answer)

    workflow.add_edge(START, "init_groq_model")
    workflow.add_edge("init_groq_model", "prepare_vector_database")
    workflow.add_edge("prepare_vector_database", "fetch_relevant_docs")
    workflow.add_edge("fetch_relevant_docs", "filter_docs_by_relevance")
    workflow.add_conditional_edges(
        "filter_docs_by_relevance",
        should_generate_answer,
        {"answer": "produce_answer", "stop": END}
    )
    workflow.add_edge("produce_answer", "detect_hallucination")
    workflow.add_edge("detect_hallucination", "validate_answer")

    return workflow.compile()


# Run workflow
if __name__ == "__main__":
    pipeline = create_workflow()
    
    result = pipeline.invoke({"user_question": "what is disjoint data structure?"})

    print("\n---FINAL OUTPUT---")
    if result.get("has_hallucination"):
        print(" Warning: The answer may contain hallucinations.\n")
    if not result.get("is_valid_answer"):
        print(" Warning: The answer may not fully address the question.\n")
    print("Answer:\n", result.get("answer_draft", "No answer generated."))
