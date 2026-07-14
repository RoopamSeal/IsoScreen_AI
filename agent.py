"""
===========================================================
GraphDrugPred / IsoScreenAI
LangGraph Workflow
===========================================================

Workflow

FASTA
   │
Validate
   │
Embedding
   │
Prediction
   │
Structural Metrics
   │
LLM Report
   │
Return Final State
"""

import re
from typing import TypedDict, List, Optional
import numpy as np

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config


# ==========================================================
# GRAPH STATE
# ==========================================================

class AgentState(TypedDict):
    fasta_content: str
    sequence: str
    embedding: Optional[List[float]]
    prediction: Optional[float]
    confidence: Optional[float]
    evolutionary_conservation: Optional[float]
    predicted_disorder: Optional[float]
    binding_free_energy: Optional[float]
    mean_plddt: Optional[float]
    pae_matrix: Optional[List[List[float]]]
    report: Optional[str]
    errors: List[str]


# ==========================================================
# GRAPH FACTORY
# ==========================================================

def create_graph(predictor):
    """
    Factory function that injects the loaded ProteinPredictor instance
    into the LangGraph execution workflow.
    """
    llm = ChatGroq(
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )

    # ======================================================
    # FASTA VALIDATION
    # ======================================================

    def validate_fasta(state: AgentState):
        errors = list(state.get("errors", []))
        fasta = state.get("fasta_content", "").strip()

        if not fasta:
            errors.append("No FASTA sequence supplied.")
            return {**state, "errors": errors}

        sequence_lines = []
        for line in fasta.splitlines():
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            cleaned = re.sub(r"[\s\d]", "", line.upper())
            sequence_lines.append(cleaned)

        sequence = "".join(sequence_lines)

        if len(sequence) == 0:
            errors.append("Unable to extract sequence.")
            return {**state, "errors": errors}

        invalid = set(sequence) - config.VALID_AMINO_ACIDS
        if invalid:
            errors.append(
                f"Invalid amino acids detected: {', '.join(sorted(invalid))}"
            )

        return {
            **state,
            "sequence": sequence,
            "errors": errors,
        }

    # ======================================================
    # EMBEDDINGS
    # ======================================================

    def extract_embeddings(state: AgentState):
        if state.get("errors"):
            return state

        try:
            embedding = predictor.get_embedding(state["sequence"])
            return {
                **state,
                "embedding": embedding.tolist()
            }
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Embedding error: {e}"]
            }

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict_druggability(state: AgentState):
        if state.get("errors"):
            return state

        try:
            embedding = np.asarray(state["embedding"], dtype=np.float32)
            probability = predictor.predict(embedding)
            confidence = abs(probability - config.DRUGGABILITY_THRESHOLD) * 2

            return {
                **state,
                "prediction": probability,
                "confidence": confidence,
            }
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Prediction error: {e}"]
            }

    # ======================================================
    # STRUCTURAL METRICS (SIMULATED FOR V2)
    # ======================================================

    def calculate_structural_metrics(state: AgentState):
        if state.get("errors"):
            return state

        seq_len = len(state["sequence"])
        np.random.seed(seq_len)

        matrix_size = min(seq_len, config.PAE_MAX_SIZE)

        return {
            **state,
            "evolutionary_conservation": float(np.random.uniform(0.65, 0.95)),
            "predicted_disorder": float(np.random.uniform(5, 25)),
            "binding_free_energy": float(np.random.uniform(-9.5, -5.0)),
            "mean_plddt": float(np.random.uniform(70, 95)),
            "pae_matrix": np.random.uniform(
                0, 30, (matrix_size, matrix_size)
            ).tolist(),
        }

    # ======================================================
    # REPORT
    # ======================================================

    def generate_report(state: AgentState):
        if state.get("errors"):
            return state

        if not config.ENABLE_REPORT_GENERATION:
            return {
                **state,
                "report": "LLM report generation disabled by system config."
            }

        verdict = (
            "Highly Druggable"
            if state["prediction"] >= config.DRUGGABILITY_THRESHOLD
            else "Poorly Druggable"
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a senior computational biologist.
Write a concise scientific target assessment report.

--- Target Summary ---
Sequence Length: {length} Residues
Druggability Probability: {prediction:.4f}
ML Model Confidence: {confidence:.4f}
Classification Verdict: {verdict}
Mean Backbone Confidence (pLDDT): {plddt:.1f}
Predicted Structural Disorder: {disorder:.1f}%
Binding Free Energy (ΔG): {dg:.2f} kcal/mol
Evolutionary Conservation Index: {conservation:.2f}

Produce a structured response with these exact markdown headers:
### 1. Executive Summary
### 2. Biological & Structural Interpretation
### 3. Druggability Assessment
### 4. Recommended Wet-Lab Experiments
"""
        )

        try:
            chain = prompt | llm
            response = chain.invoke({
                "length": len(state["sequence"]),
                "prediction": state["prediction"],
                "confidence": state["confidence"],
                "verdict": verdict,
                "plddt": state["mean_plddt"],
                "disorder": state["predicted_disorder"],
                "dg": state["binding_free_energy"],
                "conservation": state["evolutionary_conservation"]
            })

            return {
                **state,
                "report": response.content
            }
        except Exception as e:
            fallback_text = (
                f"### Automated Target Summary (LLM Fallback Mode)\n"
                f"- **Classification**: {verdict}\n"
                f"- **Druggability Probability**: {state['prediction']:.4f}\n"
                f"- **Model Confidence**: {state['confidence']:.4f}\n"
                f"- **Simulated Binding Energy**: {state['binding_free_energy']:.2f} kcal/mol\n"
            )
            return {
                **state,
                "report": fallback_text,
                "errors": state.get("errors", []) + [f"Reporting error: {str(e)}"]
            }

    # ======================================================
    # ROUTER
    # ======================================================

    def router(state: AgentState):
        if state.get("errors"):
            return "error"
        return "continue"

    # ======================================================
    # BUILD GRAPH
    # ======================================================

    graph = StateGraph(AgentState)

    graph.add_node("validate_fasta", validate_fasta)
    graph.add_node("extract_embeddings", extract_embeddings)
    graph.add_node("predict_druggability", predict_druggability)
    graph.add_node("calculate_structural_metrics", calculate_structural_metrics)
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "validate_fasta")

    graph.add_conditional_edges(
        "validate_fasta",
        router,
        {"continue": "extract_embeddings", "error": END}
    )

    graph.add_conditional_edges(
        "extract_embeddings",
        router,
        {"continue": "predict_druggability", "error": END}
    )

    graph.add_conditional_edges(
        "predict_druggability",
        router,
        {"continue": "calculate_structural_metrics", "error": END}
    )

    graph.add_conditional_edges(
        "calculate_structural_metrics",
        router,
        {"continue": "generate_report", "error": END}
    )

    graph.add_edge("generate_report", END)

    return graph.compile()
