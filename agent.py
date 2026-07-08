import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import config
from predictor import ProteinPredictor

# Lazy instantiate modeling infrastructure
predictor_instance = ProteinPredictor()

class AgentState(TypedDict):
    fasta_content: str
    sequence: str
    embedding: Optional[List[float]]
    prediction: Optional[float]
    confidence: Optional[float]
    report: Optional[str]
    errors: List[str]

def validate_fasta(state: AgentState) -> AgentState:
    """Sanitizes raw unstructured inputs down to structured standard AA characters."""
    errors = list(state.get("errors", []))
    fasta = state.get("fasta_content", "").strip()

    if not fasta:
        errors.append("Validation Error: Input payload was parsed empty.")
        return {**state, "errors": errors}

    lines = fasta.split("\n")
    sequence_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        # Strip illegal characters, numerical markers, and white spaces
        cleaned = re.sub(r'[\s\d]', '', line).upper()
        sequence_lines.append(cleaned)

    full_sequence = "".join(sequence_lines)
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    unrecognized_residues = set(full_sequence) - canonical_amino_acids

    if unrecognized_residues:
        errors.append(f"Validation Error: Contaminant tokens detected: {', '.join(unrecognized_residues)}")
        return {**state, "errors": errors}
        
    if not full_sequence:
        errors.append("Validation Error: Failed to extract target residue rows.")
        return {**state, "errors": errors}

    return {**state, "sequence": full_sequence, "errors": errors}

def extract_embeddings(state: AgentState) -> AgentState:
    """Transforms raw verified strings into standardized model feature spaces."""
    if state.get("errors"):
        return state
    try:
        embedding_array = predictor_instance.get_embedding(state["sequence"])
        return {**state, "embedding": embedding_array.tolist()}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Embedding Extraction Failure: {str(e)}")
        return {**state, "errors": errors}

def predict_druggability(state: AgentState) -> AgentState:
    """Interrogates classical layers using extracted target protein vectors."""
    if state.get("errors"):
        return state
    try:
        import numpy as np
        vector = np.array(state["embedding"])
        probability = predictor_instance.predict(vector)
        
        # Calculate standard deviation margin scale mapped relative to classification threshold
        confidence = abs(probability - config.DRUGGABILITY_THRESHOLD) * 2
        return {**state, "prediction": probability, "confidence": confidence}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Prediction Pipeline Failure: {str(e)}")
        return {**state, "errors": errors}

def generate_report(state: AgentState) -> AgentState:
    """Constructs academic summary insights via specialized remote LLM agents."""
    if state.get("errors"):
        return state
    try:
        # Pulls GOOGLE_API_KEY from environment contexts automatically
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
        score = state["prediction"]
        verdict = "Druggable" if score >= config.DRUGGABILITY_THRESHOLD else "Non-Druggable"
        
        prompt = ChatPromptTemplate.from_template(
            "You are acting as a Senior Principal Bioinformatician specializing in target assessment.\n"
            "Review the quantitative indicators generated for this sequence sample:\n\n"
            "- Total Sequence Length: {seq_len} Residues\n"
            "- Target Druggability Likelihood: {score:.4f}\n"
            "- Status Classification: {verdict}\n"
            "- Agent Structural Confidence: {confidence:.4f}\n\n"
            "Compose a concise, academic summary report divided explicitly into:\n"
            "### 1. Executive Screening Assessment\n"
            "### 2. Physical & Translation Insights\n"
            "### 3. Druggability Mechanism Deductions\n"
            "### 4. Downstream Assay Recommendations\n\n"
            "Maintain professional, expert, objective domain jargon throughout."
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "seq_len": len(state["sequence"]),
            "score": score,
            "verdict": verdict,
            "confidence": state["confidence"]
        })
        return {**state, "report": response.content}
        
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Reporting Subsystem Warning: {str(e)}")
        fallback = (
            f"### Automated Summary Report (Fallback Mode)\n"
            f"- **Classification Output**: {'Druggable' if state['prediction'] >= config.DRUGGABILITY_THRESHOLD else 'Non-Druggable'}\n"
            f"- **Scoring Profile**: {state['prediction']:.4f}\n"
            f"- **System Assurance Level**: {state['confidence']:.4f}\n"
        )
        return {**state, "report": fallback, "errors": errors}

# Workflow Graph Assembly
builder = StateGraph(AgentState)
builder.add_node("validate_fasta", validate_fasta)
builder.add_node("extract_embeddings", extract_embeddings)
builder.add_node("predict_druggability", predict_druggability)
builder.add_node("generate_report", generate_report)

builder.add_edge(START, "validate_fasta")
builder.add_edge("validate_fasta", "extract_embeddings")
builder.add_edge("extract_embeddings", "predict_druggability")
builder.add_edge("predict_druggability", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
