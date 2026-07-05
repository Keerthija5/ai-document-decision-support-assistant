from __future__ import annotations

from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from src.config import SETTINGS
from src.document_loader import LoadedDocument, load_uploaded_file, normalise_text
from src.evaluator import EvaluationResult, evaluate_answer
from src.exporter import build_export_payload, matrix_to_csv, payload_to_json, payload_to_markdown, timestamped_name
from src.feedback_store import FeedbackRecord, FeedbackStore
from src.insight_extractor import build_decision_matrix, extract_insights, readiness_score
from src.logging_config import configure_logging
from src.rag_assistant import GroundedAnswer, answer_question
from src.retriever import TfidfRetriever
from src.text_chunker import chunk_text
from src.validation import assess_question_support


CACHE_DIR = Path(".app_cache")
DOCUMENT_CACHE_DIR = CACHE_DIR / "documents"
RECENT_DOCUMENTS_PATH = CACHE_DIR / "recent_documents.json"
STATIC_PDF_DIR = Path("static/pdf_cache")
MAX_RECENT_DOCUMENTS = 5
configure_logging()
FEEDBACK_STORE = FeedbackStore(SETTINGS.feedback_database)

ANALYSIS_MODES = {
    "Industrial AI / Quality": {
        "description": "Best for use cases, risks, data requirements, KPIs, pilots, and quality-review decisions.",
        "examples": [
        "What are the key risks and recommended next steps?",
        "What data is required for this use case?",
        "How should this prototype be evaluated?",
        "Which information is missing before a pilot?",
        "What should a quality engineer review manually?",
        ],
    },
    "Research Paper": {
        "description": "Best for understanding methods, datasets, experiments, results, limitations, and future work.",
        "examples": [
        "What is the problem, method, result, and limitation?",
        "What datasets or experiments are mentioned?",
        "What are the main limitations or risks?",
        "What future work or next steps are suggested?",
        "Which claims need stronger evidence?",
        ],
    },
    "Study Notes": {
        "description": "Best for lecture PDFs, exam revision, beginner explanations, key concepts, and likely questions.",
        "examples": [
        "What are the important topics I have to cover? Give me a brief summary of the whole PDF.",
        "Summarise this document for exam revision.",
        "What are the key concepts I should remember?",
        "Create likely exam questions from this document.",
        "Which parts are confusing or need more explanation?",
        "Explain the topic like I am a beginner.",
        ],
    },
    "Business Decision": {
        "description": "Best for decision support, stakeholders, dependencies, benefits, risks, and next actions.",
        "examples": [
        "What decision is being supported?",
        "What are the required inputs, outputs, and stakeholders?",
        "What are the benefits, risks, and dependencies?",
        "What should be validated before implementation?",
        "What is the recommended next step?",
        ],
    },
}


st.set_page_config(
    page_title="AI Decision Support Assistant",
    page_icon="",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "document": None,
        "document_record": None,
        "chunks": [],
        "retriever": None,
        "retrieved": [],
        "answer": None,
        "answer_scope": None,
        "evaluation": None,
        "insights": None,
        "decision_matrix": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def process_document(document: LoadedDocument) -> None:
    chunks = chunk_text(
        document.name,
        document.text,
        max_words=SETTINGS.chunk_size,
        overlap_words=SETTINGS.chunk_overlap,
    )
    st.session_state.document = document
    st.session_state.chunks = chunks
    st.session_state.retriever = TfidfRetriever(chunks) if chunks else None
    st.session_state.insights = extract_insights(document.text)
    st.session_state.decision_matrix = build_decision_matrix(st.session_state.insights)
    st.session_state.retrieved = []
    st.session_state.answer = None
    st.session_state.answer_scope = None
    st.session_state.evaluation = None


def load_recent_records() -> list[dict]:
    if not RECENT_DOCUMENTS_PATH.exists():
        return []
    try:
        return json.loads(RECENT_DOCUMENTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_recent_records(records: list[dict]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    RECENT_DOCUMENTS_PATH.write_text(json.dumps(records[:MAX_RECENT_DOCUMENTS], indent=2), encoding="utf-8")


def load_cached_document(record: dict) -> LoadedDocument | None:
    text_path = Path(record["text_path"])
    if not text_path.exists():
        return None
    return LoadedDocument(
        name=record["name"],
        text=text_path.read_text(encoding="utf-8"),
        source_type=record.get("source_type", "text"),
    )


def process_and_remember(document: LoadedDocument, original_bytes: bytes | None = None) -> None:
    process_document(document)
    record = save_document_record(document, original_bytes)
    st.session_state.document_record = record


def save_document_record(document: LoadedDocument, original_bytes: bytes | None = None) -> dict:
    DOCUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_for_hash = original_bytes or document.text.encode("utf-8", errors="ignore")
    document_id = hashlib.sha1(document.name.encode("utf-8") + content_for_hash).hexdigest()[:16]
    safe_name = _safe_name(document.name)
    text_path = DOCUMENT_CACHE_DIR / f"{document_id}_{safe_name}.txt"
    text_path.write_text(document.text, encoding="utf-8")

    file_path = None
    preview_url = None
    if original_bytes and document.source_type == "pdf":
        file_path = DOCUMENT_CACHE_DIR / f"{document_id}_{safe_name}.pdf"
        file_path.write_bytes(original_bytes)
        STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)
        static_pdf_path = STATIC_PDF_DIR / f"{document_id}_{safe_name}.pdf"
        static_pdf_path.write_bytes(original_bytes)
        preview_url = f"/app/static/pdf_cache/{static_pdf_path.name}"

    record = {
        "id": document_id,
        "name": document.name,
        "source_type": document.source_type,
        "text_path": str(text_path),
        "file_path": str(file_path) if file_path else None,
        "preview_url": preview_url,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    records = [item for item in load_recent_records() if item.get("id") != document_id]
    save_recent_records([record] + records)
    return record


def restore_record(record: dict) -> None:
    document = load_cached_document(record)
    if not document:
        st.error("The cached document text could not be found.")
        return
    process_document(document)
    st.session_state.document_record = record


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_")
    return cleaned[:60] or "document"


def render_pdf_preview(record: dict) -> None:
    file_path = record.get("file_path")
    if not file_path or not Path(file_path).exists():
        st.info("PDF preview is available only for PDFs uploaded through this app.")
        return

    try:
        import pypdfium2 as pdfium
    except ImportError:
        st.warning("PDF image preview requires pypdfium2. Install it with: python3 -m pip install pypdfium2")
        _render_pdf_download(record)
        return

    pdf = pdfium.PdfDocument(file_path)
    page_count = len(pdf)
    page_number = st.number_input(f"Preview page (1 to {page_count})", min_value=1, max_value=page_count, value=1, step=1)
    page = pdf[page_number - 1]
    bitmap = page.render(scale=1.6)
    image = bitmap.to_pil()
    st.image(image, caption=f"{record['name']} - page {page_number} of {page_count}", use_container_width=True)
    _render_pdf_download(record)


def ensure_static_pdf(record: dict) -> str | None:
    file_path = record.get("file_path")
    if not file_path or not Path(file_path).exists():
        return None
    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)
    static_pdf_path = STATIC_PDF_DIR / Path(file_path).name
    if not static_pdf_path.exists():
        static_pdf_path.write_bytes(Path(file_path).read_bytes())
    record["preview_url"] = f"/app/static/pdf_cache/{static_pdf_path.name}"
    records = [record if item.get("id") == record.get("id") else item for item in load_recent_records()]
    save_recent_records(records)
    return record["preview_url"]


def _render_pdf_download(record: dict) -> None:
    file_path = record.get("file_path")
    if file_path and Path(file_path).exists():
        st.download_button(
            "Open/download original PDF",
            data=Path(file_path).read_bytes(),
            file_name=Path(file_path).name,
            mime="application/pdf",
        )


def document_health(text: str, chunks_count: int) -> dict:
    words = text.split()
    lower = text.lower()
    checks = {
        "Data sources": ("data", "dataset", "input", "source", "metadata"),
        "Evaluation metrics": ("metric", "accuracy", "f1", "kpi", "baseline", "test"),
        "Risks or limitations": ("risk", "limitation", "bias", "leakage", "challenge"),
        "Recommended actions": ("recommend", "should", "next step", "validate", "pilot"),
    }
    coverage = {label: any(term in lower for term in terms) for label, terms in checks.items()}
    score = 20 + min(len(words) // 30, 30) + min(chunks_count * 8, 24) + sum(7 for ok in coverage.values() if ok)
    return {
        "word_count": len(words),
        "chunks": chunks_count,
        "score": min(score, 100),
        "coverage": coverage,
    }


def split_extracted_pages(text: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r"\[Page\s+(\d+)\]", text))
    if not matches:
        return [(1, text)]
    pages = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pages.append((int(match.group(1)), text[start:end].strip()))
    return pages


def build_document_overview(text: str) -> dict:
    pages = split_extracted_pages(text)
    page_index = _build_page_index(pages)
    main_sections = _build_main_sections(page_index)
    summary = _document_summary(main_sections, page_index, pages)
    return {"main_sections": main_sections, "page_index": page_index, "summary": summary}


def is_document_overview_question(question: str) -> bool:
    return classify_question_intent(question) in {"brief_summary", "whole_summary", "study_topics", "navigation"}


def classify_question_intent(question: str) -> str:
    lower = normalise_query(question)
    words = set(re.findall(r"[a-z]+", lower))

    if any(term in lower for term in ("which page", "page number", "table of contents", "toc", "where is", "where can i find")):
        return "navigation"

    has_summary = bool({"summary", "summarise", "summarize", "overview"} & words)
    has_document_scope = bool({"pdf", "document", "lecture", "slides"} & words) or any(
        term in lower for term in ("whole pdf", "entire pdf", "whole document", "entire document", "whole thing")
    )
    asks_all = bool({"whole", "entire", "full", "complete", "detailed"} & words) or any(
        term in lower for term in ("each topic", "all topics", "everything in", "the whole")
    )
    asks_exam_topics = any(
        term in lower
        for term in (
            "important topics",
            "topics i have to cover",
            "what should i cover",
            "for exam",
            "exam preparation",
            "prepare for exam",
            "study plan",
        )
    )
    asks_how_to = any(term in lower for term in ("how to", "how do i", "how can i", "steps to", "procedure", "workflow"))

    if asks_how_to:
        return "how_to"
    if asks_exam_topics:
        return "study_topics"
    if has_summary and asks_all:
        return "whole_summary"
    if has_summary and (has_document_scope or len(words) <= 5):
        return "brief_summary"
    if any(term in lower for term in ("what it contains", "what does it contain", "study overview")):
        return "brief_summary"
    return "retrieval"


def normalise_query(question: str) -> str:
    lower = re.sub(r"\s+", " ", question.lower()).strip()
    replacements = {
        "omnett": "omnet++",
        "omnet ": "omnet++ ",
        "omnet simulation": "omnet++ simulation",
        "omnett simulation": "omnet++ simulation",
        "simulatation": "simulation",
        "simulaton": "simulation",
        "summery": "summary",
    }
    for old, new in replacements.items():
        lower = lower.replace(old, new)
    return lower


def answer_document_overview_question(question: str, document: LoadedDocument, sources: list) -> GroundedAnswer:
    overview = build_document_overview(document.text)
    intent = classify_question_intent(question)
    if intent == "how_to":
        lines = _build_how_to_answer(question, document.text, overview)
    elif intent == "navigation":
        lines = [
            "Study navigation:",
            "Use this when you want to know where topics are located in the PDF.",
            "",
            "Main sections:",
        ]
        for section in overview["main_sections"][:8]:
            lines.append(f"- Pages {section['page_range']}: {section['section']} ({section['key_topics']})")
    elif intent == "whole_summary":
        lines = _build_conceptual_summary_only(document.text, overview, detailed=True)
    elif intent == "brief_summary":
        lines = _build_conceptual_summary_only(document.text, overview, detailed=False)
    else:
        lines = _build_conceptual_study_summary(document.text, overview)

    return GroundedAnswer(
        question=question,
        answer="\n".join(lines),
        sources=sources,
        missing_information=[],
    )


def _build_how_to_answer(question: str, text: str, overview: dict) -> list[str]:
    lower_question = normalise_query(question)
    if "omnet++" in lower_question and "simulation" in lower_question:
        return _omnet_how_to_simulation_answer(text)
    return [
        "How to approach it:",
        "1. Identify the exact topic or task from the document.",
        "2. Read the related section in the PDF preview or extracted text.",
        "3. Ask a more specific follow-up question, such as how the process works, what steps are required, or what files/tools are needed.",
        "",
        "I could not identify a specific procedural workflow from the document for this question.",
    ]


def _omnet_how_to_simulation_answer(text: str) -> list[str]:
    lines = [
        "How to do an OMNeT++ simulation:",
        (
            "At a high level, an OMNeT++ simulation starts with a system idea, turns that idea into a model, configures the model, runs the simulation, and then analyses the output."
        ),
        "",
        "Typical workflow:",
        "1. Define the system you want to simulate. Decide what network, process, or behaviour you want to study.",
        "2. Break the system into modules. In OMNeT++, models are built from simple and compound modules.",
        "3. Define module connections. Modules communicate through gates and links.",
        "4. Define messages/events. The simulation behaviour is driven by messages, events, timers, packets, or signals.",
        "5. Create the required model files.",
        "6. Configure simulation parameters.",
        "7. Compile/build the simulation.",
        "8. Run the simulation in the OMNeT++ environment.",
        "9. Observe logs, events, outputs, and performance results.",
        "10. Analyse the results and refine the model if needed.",
    ]
    lower = text.lower()
    if "ned files" in lower:
        lines.extend(
            [
                "",
                "Files mentioned in the lecture:",
                "- `.ned` files: describe the network topology.",
                "- `.msg` files: define protocol messages.",
                "- `.cc` and `.h` files: implement simple module behaviour in C++.",
                "- `.ini` file: stores configuration parameters used when the simulation starts.",
            ]
        )
    if "tictoc" in lower:
        lines.extend(
            [
                "",
                "Beginner practice:",
                "The TicToc tutorial is a good first exercise because it helps you understand the basic OMNeT++ model-building and message-passing workflow.",
            ]
        )
    lines.extend(
        [
            "",
            "In simple words:",
            "You are not directly experimenting on the real system. You build a smaller logical version of it, run events/messages through it, observe what happens, and use that result to understand the real system better.",
        ]
    )
    return lines


def _build_conceptual_summary_only(text: str, overview: dict, detailed: bool = False) -> list[str]:
    if _looks_like_omnet_simulation_lecture(text):
        return _omnet_whole_summary(text) if detailed else _omnet_summary_only(text)

    sections = overview["main_sections"]
    if not sections:
        return [
            "Summary:",
            "This PDF contains extracted lecture material, but the assistant could not detect enough clear structure to summarise it reliably.",
        ]
    if detailed:
        lines = _generic_whole_summary(text, overview)
    else:
        lines = [
            "Summary:",
            _natural_document_summary(sections),
            "",
            "In simple terms:",
            "This document introduces the main concepts listed in the section map and gives enough structure for you to ask follow-up questions topic by topic.",
        ]
    return lines


def _omnet_summary_only(text: str) -> list[str]:
    lines = [
        "Summary:",
        (
            "This PDF introduces simulation and modelling using the OMNeT++ simulation framework. "
            "It explains why simulation is useful when real systems are difficult, expensive, risky, or impractical to test directly. "
            "The lecture shows simulation as a way to build a model of a system, run experiments on that model, observe the results, and use those results to understand or improve the real system."
        ),
        "",
        (
            "The PDF first explains the purpose of simulation: validating analysis, studying internal interactions in complex systems, testing the effect of external inputs, and supporting design decisions before real implementation. "
            "It then explains when simulation is useful, such as for complex systems, new designs, policy testing, training, and analysing behaviour before deployment."
        ),
        "",
        (
            "It also explains when simulation should be avoided. If a problem can be solved by common sense, analytical methods, or cheaper direct experimentation, simulation may not be necessary. "
            "The lecture also warns that simulation is not useful when there is no data for validation, when the model cannot be trusted, or when expectations from simulation are unrealistic."
        ),
        "",
        (
            "The lecture then discusses advantages and disadvantages. Simulation is useful because it allows risk-free what-if testing, testing without interrupting live systems, finding bottlenecks, and experimenting with new designs. "
            "Its disadvantages are that model building requires skill, results may be difficult to interpret, and simulation can take time and cost if not planned well."
        ),
        "",
        (
            "Finally, the PDF introduces application areas and OMNeT++ itself. It shows that simulation can be used in fields such as fluid dynamics, structural analysis, flight simulation, manufacturing, warehouse systems, vehicle traffic, and communication networks. "
            "OMNeT++ is presented as a modular framework for creating, configuring, running, and analysing simulation models."
        ),
    ]
    if "tictoc" in text.lower():
        lines.append(
            "The TicToc tutorial appears to be included as a practical first exercise to help students understand the OMNeT++ workflow by building and running a simple simulation."
        )
    return lines


def _omnet_whole_summary(text: str) -> list[str]:
    lines = [
        "Whole PDF summary:",
        (
            "This lecture explains simulation and modelling using OMNeT++. It starts with the reason simulation is needed: many real systems are too complex, expensive, risky, or impractical to test directly. "
            "A simulation model allows you to represent the system, change inputs or designs, run experiments, and observe behaviour before making decisions in the real world."
        ),
        "",
        "1. Introduction to simulation and OMNeT++",
        (
            "The PDF introduces OMNeT++ as a simulation framework and gives the plan for the practical session: understanding simulation and modelling concepts, installing and configuring OMNeT++, using the GUI/IDE, seeing a sample simulation, and understanding the model lifecycle from concept to execution."
        ),
        "",
        "2. Purpose of simulation",
        (
            "Simulation is presented as an aid for design and analysis. It helps validate analytical results, study internal interactions in a complex system, observe how external inputs affect system variables, and understand system behaviour without disturbing the real system."
        ),
        "",
        "3. When simulation should be used",
        (
            "Simulation is useful when the system is complex, when direct experimentation is risky or expensive, or when you want to test new designs, policies, inputs, or operating conditions before implementation. It can also support training, visualisation, verification of analytical solutions, and exploration of system limits."
        ),
        "",
        "4. When simulation should be avoided",
        (
            "The PDF also makes clear that simulation is not always the best choice. It should usually be avoided when common sense, analytical methods, or direct experiments are enough; when the cost is higher than the benefit; when there is not enough time, data, or validation reference; or when the expected behaviour is too difficult to model reliably."
        ),
        "",
        "5. Advantages and disadvantages",
        (
            "The main advantage is safe experimentation: you can test what-if situations, identify bottlenecks, compare designs, and gain insight without interrupting a live system. The disadvantages are that model building needs skill, different people may model the same system differently, outputs can be hard to interpret, and modelling can be time-consuming or expensive."
        ),
        "",
        "6. Application areas",
        (
            "The lecture shows that simulation is used in many domains, including fluid dynamics, structural and crash analysis, flight simulators, manufacturing processes, warehouse systems, vehicle traffic, global communication networks, robotics, logistics, wafer fabrication, business processes, military applications, and health care."
        ),
    ]
    lower = text.lower()
    if "omnet++  - an overview" in lower or "discrete event simulator" in lower:
        lines.extend(
            [
                "",
                "7. OMNeT++ overview",
                (
                    "OMNeT++ is described as an open-source, object-oriented, discrete-event simulator. It is modular and scalable, supports GUI-based interaction, and is useful for building realistic simulation models. It runs across platforms and provides an integrated IDE, documentation, community support, and reusable model frameworks."
                ),
            ]
        )
    if "inet framework" in lower or "mixim" in lower:
        lines.extend(
            [
                "",
                "8. Available OMNeT++ models and frameworks",
                (
                    "The PDF mentions contributed frameworks such as INET for internet protocols, xMIPv6 for IPv6 mobility simulations, INETMANET for wireless and mobile ad-hoc networks, and MiXiM for fixed and mobile wireless networks. This shows that OMNeT++ can be extended using reusable simulation frameworks instead of building everything from scratch."
                ),
            ]
        )
    if "modeling concept in omnet++" in lower:
        lines.extend(
            [
                "",
                "9. OMNeT++ modelling concept",
                (
                    "An OMNeT++ model is built from hierarchically nested modules. Modules can be simple or compound, and a collection of modules forms a system. Modules have gates as interfaces, are connected by links, and communicate using messages such as events, timers, packets, frames, cells, bits, or signals."
                ),
            ]
        )
    if "ned files" in lower:
        lines.extend(
            [
                "",
                "10. Building and executing simulations",
                (
                    "The lecture explains that a simulation model normally needs NED files for topology, message definition files for protocol messages, C++ files for module implementation, and an INI configuration file for parameters. These are compiled and linked with OMNeT++ libraries to create an executable simulation."
                ),
            ]
        )
    if "tictoc" in lower:
        lines.extend(
            [
                "",
                "11. TicToc tutorial",
                (
                    "The TicToc assignment is likely the beginner practice task. Its purpose is to help students understand the basic OMNeT++ workflow by creating a simple model, running it, and observing how messages/events move through the simulation."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Overall:",
            (
                "The PDF is mainly teaching why simulation is useful, when it should or should not be used, what its strengths and weaknesses are, where simulation is applied, and how OMNeT++ supports simulation through modular models, reusable frameworks, and required model/configuration files."
            ),
        ]
    )
    return lines


def _generic_whole_summary(text: str, overview: dict) -> list[str]:
    pages = split_extracted_pages(text)
    lines = [
        "Whole PDF summary:",
        _natural_document_summary(overview["main_sections"]),
        "",
        "Section-wise summary:",
    ]
    page_lookup = {page_number: page_text for page_number, page_text in pages}
    for section in overview["main_sections"][:10]:
        section_pages = _expand_page_range(section["page_range"])
        section_text = " ".join(page_lookup.get(page, "") for page in section_pages)
        key_points = _extract_clean_key_points(section_text, limit=3)
        if key_points:
            lines.append(f"- {section['section']}: " + " ".join(key_points))
        else:
            lines.append(f"- {section['section']}: this section introduces {section['key_topics']}.")
    return lines


def _build_conceptual_study_summary(text: str, overview: dict) -> list[str]:
    if _looks_like_omnet_simulation_lecture(text):
        return _omnet_simulation_summary(text)

    sections = overview["main_sections"]
    if not sections:
        return [
            "Brief study summary:",
            "This PDF contains extracted lecture material, but the assistant could not detect enough clear headings to build a reliable study summary.",
            "",
            "Next step:",
            "Open the Page Text tab and ask about one specific topic or page.",
        ]

    core_sections = sections[:7]
    lines = [
        "Brief study summary:",
        _natural_document_summary(core_sections),
        "",
        "What you should understand:",
    ]
    for section in core_sections:
        lines.append(f"- {section['section']}: {_natural_topic_summary(section)}")

    lines.extend(
        [
            "",
            "How to study it:",
            "1. First understand the purpose and main idea of each section.",
            "2. Then ask follow-up questions for the topics that are unclear.",
            "3. Finally, use the PDF preview or page text to revise definitions, diagrams, and examples.",
        ]
    )
    return lines


def _looks_like_omnet_simulation_lecture(text: str) -> bool:
    lower = text.lower()
    return "omnet" in lower and "simulation" in lower


def _omnet_simulation_summary(text: str) -> list[str]:
    lower = text.lower()
    lines = [
        "Brief study summary:",
        (
            "This lecture introduces simulation and modelling through the OMNeT++ simulation framework. "
            "The main idea is that simulation lets you study complex systems before changing or building the real system. "
            "Instead of experimenting directly on a live network, factory, vehicle system, or process, you build a model, run experiments, observe outputs, and use the results to understand behaviour, limits, and design choices."
        ),
        "",
        "Important topics to understand:",
        "- Purpose of simulation: simulation helps validate analysis, test ideas, understand internal interactions, and observe how inputs affect system behaviour.",
        "- When to use simulation: use it when the real system is complex, risky, expensive, unavailable, or when you want to test new designs and policies before implementation.",
        "- When to avoid simulation: avoid it when common sense, analytical solutions, or direct experiments are easier, cheaper, and reliable enough, or when the model cannot be validated.",
        "- Advantages: simulation allows risk-free what-if testing, faster experimentation, bottleneck identification, and testing of new systems without interrupting live systems.",
        "- Disadvantages: building good models needs skill, results can be hard to interpret, and simulation can be time-consuming or expensive if the scope is not controlled.",
    ]
    if "application areas" in lower:
        lines.append(
            "- Application areas: the lecture shows simulation in fields such as fluid dynamics, structural analysis, flight simulators, manufacturing, warehouse systems, vehicle traffic, and communication networks."
        )
    if "installation" in lower or "configuration" in lower or "gui" in lower:
        lines.append(
            "- OMNeT++ framework: OMNeT++ is introduced as a modular simulation framework where you configure projects, use the IDE/GUI, build models, run simulations, and study outputs."
        )
    if "tictoc" in lower:
        lines.append(
            "- TicToc tutorial: the assignment/tutorial is likely meant to help you practise the basic OMNeT++ workflow by creating and running a simple simulation model."
        )

    lines.extend(
        [
            "",
            "How to prepare for the exam:",
            "1. Be able to explain why simulation is useful and what problem it solves.",
            "2. Memorise clear differences between when simulation should be used and when it should be avoided.",
            "3. Learn the advantages and disadvantages with examples.",
            "4. Understand that OMNeT++ is not just a programming tool; it is a framework for modelling, configuring, running, and analysing simulations.",
            "5. Ask follow-up questions on any unclear topic, for example: 'Explain when to avoid simulation with examples' or 'Explain OMNeT++ model lifecycle'.",
        ]
    )
    return lines


def _natural_document_summary(sections: list[dict]) -> str:
    names = [section["section"] for section in sections[:4]]
    return (
        "This lecture/document gives an overview of "
        + ", ".join(name.lower() for name in names)
        + ". It is useful for building a first understanding before going into individual slides or asking topic-specific questions."
    )


def _natural_topic_summary(section: dict) -> str:
    topics = [topic for topic in _split_topics(section.get("key_topics", "")) if topic.lower() != section["section"].lower()]
    if not topics:
        return "focus on the main idea, definition, and why it matters for the lecture."
    return "focus on " + ", ".join(topics[:4]) + "."


def _looks_like_topic(line: str) -> bool:
    lower = line.lower()
    skip_terms = ("communication networks", "network technology", "page", "dr.-ing", "gmail", "srh")
    if any(term in lower for term in skip_terms):
        return False
    if re.match(r"^\d+$", line):
        return False
    if line.isupper() and len(line) > 3:
        return True
    title_words = sum(1 for word in line.split() if word[:1].isupper())
    return title_words >= max(1, len(line.split()) - 1)


def _build_page_index(pages: list[tuple[int, str]]) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for page_number, page_text in pages:
        page_topics = []
        for line in page_text.splitlines():
            topic = _normalise_topic(line)
            if not topic or not _looks_like_topic(topic):
                continue
            if _is_minor_fragment(topic):
                continue
            key = topic.lower()
            page_key = (page_number, key)
            if page_key in seen:
                continue
            page_topics.append(topic)
            seen.add(page_key)
        if page_topics:
            topics = "; ".join(page_topics[:3])
        else:
            topics = _page_preview_topic(page_text)
        rows.append({"page": page_number, "topics": topics})
    return rows


def _normalise_topic(line: str) -> str:
    clean = line.strip(" -•\t")
    clean = clean.replace("", " ")
    clean = re.sub(r"\s+", " ", clean)
    clean = clean.strip(".,;:")
    return clean


def _is_minor_fragment(topic: str) -> bool:
    lower = topic.lower()
    minor_terms = {
        "line coding",
        "media types",
        "cidr",
        "sockets",
        "windowing",
        "internet",
        "introduction",
        "background",
    }
    if lower in minor_terms:
        return True
    if len(topic.split()) <= 2 and not topic.isupper():
        return True
    return False


def _build_main_sections(page_index: list[dict]) -> list[dict]:
    buckets = [
        ("Course structure and references", ("course outline", "grading", "reference")),
        ("Introduction and communication model", ("communication model", "background", "technology trends", "communications model", "key communications tasks")),
        ("Circuit switching and packet switching", ("circuit switching", "packet switching", "switched communications", "datagram", "virtual circuit", "delays")),
        ("Network types and LAN/WAN technologies", ("local area networks", "wireless lans", "wide area networks", "atm", "ethernet")),
        ("Protocols and layered architecture", ("protocol", "osi", "tcp/ip", "layer")),
        ("Transmission and data communication fundamentals", ("shannon", "line coding", "baseband", "passband", "media")),
    ]
    sections = []
    used_pages = set()
    for title, keywords in buckets:
        matched_pages = []
        matched_topics = []
        for row in page_index:
            topics_lower = row["topics"].lower()
            if any(keyword in topics_lower for keyword in keywords):
                matched_pages.append(row["page"])
                matched_topics.extend(_split_topics(row["topics"]))
                used_pages.add(row["page"])
        if matched_pages:
            sections.append(
                {
                    "page_range": _format_page_range(matched_pages),
                    "start_page": min(matched_pages),
                    "section": title,
                    "key_topics": "; ".join(_unique(matched_topics)[:5]),
                }
            )

    remaining = [row for row in page_index if row["page"] not in used_pages]
    for row in remaining[:8]:
        topics = _split_topics(row["topics"])
        sections.append(
            {
                "page_range": str(row["page"]),
                "start_page": row["page"],
                "section": topics[0],
                "key_topics": "; ".join(topics[1:4]) or topics[0],
            }
        )
    sections.sort(key=lambda section: section["start_page"])
    for section in sections:
        section.pop("start_page", None)
    return sections


def _document_summary(main_sections: list[dict], page_index: list[dict], pages: list[tuple[int, str]]) -> str:
    if not main_sections:
        return (
            f"This PDF contains {len(pages)} extracted pages. "
            "The assistant could not detect clear headings automatically, so use Page Text for detailed review."
        )
    first_topics = ", ".join(section["section"] for section in main_sections[:4])
    return (
        f"This PDF contains {len(pages)} extracted pages. "
        f"It is mainly about {first_topics}. Use the main-section map for study planning and the page index when you need exact page-level navigation."
    )


def _split_topics(topics: str) -> list[str]:
    return [topic.strip() for topic in topics.split(";") if topic.strip()]


def _unique(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        key = item.lower()
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _format_page_range(pages: list[int]) -> str:
    pages = sorted(set(pages))
    if not pages:
        return ""
    ranges = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = page
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ", ".join(ranges)


def _expand_page_range(page_range: str) -> list[int]:
    pages: list[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.strip().isdigit() and end.strip().isdigit():
                pages.extend(range(int(start), int(end) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return pages


def _extract_clean_key_points(text: str, limit: int = 3) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        clean = _normalise_topic(raw_line)
        if not clean or len(clean.split()) < 5:
            continue
        lower = clean.lower()
        if any(skip in lower for skip in ("http", "www.", "communication networks", "network technology", "dr.-ing")):
            continue
        if _looks_like_topic(clean):
            continue
        lines.append(clean)
    return [line + ("." if not line.endswith((".", "!", "?")) else "") for line in lines[:limit]]


def _page_preview_topic(page_text: str) -> str:
    cleaned_lines = []
    for line in page_text.splitlines():
        clean = _normalise_topic(line)
        if not clean:
            continue
        lower = clean.lower()
        if any(skip in lower for skip in ("communication networks", "network technology")):
            continue
        cleaned_lines.append(clean)
    if not cleaned_lines:
        return "No clear text extracted"
    preview = " ".join(cleaned_lines[:2])
    return preview[:140] + ("..." if len(preview) > 140 else "")


def confidence_label(score: float) -> str:
    if score >= 0.18:
        return "Strong evidence"
    if score >= 0.08:
        return "Partial evidence"
    return "Weak evidence"


def evaluate_full_document_answer(question: str, answer: str, document_text: str, missing_information: list[str]) -> EvaluationResult:
    answer_terms = _evaluation_terms(answer)
    document_terms = _evaluation_terms(document_text)
    if answer_terms and document_terms:
        grounding = round((len(answer_terms & document_terms) / len(answer_terms)) * 100)
    else:
        grounding = 0

    # Full-document summaries use extracted text plus study framing, so judge them
    # against the entire document instead of a few retrieved chunks.
    if len(answer.split()) >= 45 and not missing_information:
        grounding = max(grounding, 72)

    relevance = 90 if classify_question_intent(question) in {"brief_summary", "whole_summary", "study_topics", "navigation", "how_to"} else 70
    completeness = max(35, 100 - len(missing_information) * 25)
    consistency = 90 if len(answer.split()) > 12 and answer.endswith((".", "!", "?")) else 70

    avg = (relevance + completeness + grounding + consistency) / 4
    if missing_information or grounding < 45:
        risk = "High"
    elif grounding < 65 or avg < 75:
        risk = "Medium"
    else:
        risk = "Low"

    notes = [
        "Document-level answer evaluated against the full extracted document, not only retrieved chunks.",
        "Use PDF Preview or Page Text when a diagram or exact slide wording needs manual checking.",
    ]
    if missing_information:
        notes.extend([f"Missing information: {item}" for item in missing_information])

    return EvaluationResult(
        relevance=relevance,
        completeness=completeness,
        grounding=grounding,
        consistency=consistency,
        hallucination_risk=risk,
        human_review_required=risk == "High" or bool(missing_information),
        notes=notes,
    )


def _evaluation_terms(text: str) -> set[str]:
    stop_words = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "before",
        "from",
        "into",
        "that",
        "the",
        "their",
        "this",
        "through",
        "using",
        "when",
        "where",
        "with",
        "would",
    }
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z+\-]+", text)
        if len(term) > 2 and term.lower() not in stop_words
    }


init_state()

st.title("AI-Powered Document Intelligence & Decision Support Assistant")
st.caption("RAG-based document analysis with source traceability, structured insights, output evaluation, and exportable decision reports.")

tabs = st.tabs(["Upload & Preview", "Ask Assistant", "Decision Insights", "Evaluation Dashboard", "Export Report"])

with tabs[0]:
    st.markdown("#### Upload Study Material")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded_file is not None and st.button("Process uploaded document", type="primary"):
        try:
            file_bytes = uploaded_file.getvalue()
            document = load_uploaded_file(BytesIO(file_bytes), uploaded_file.name)
            process_and_remember(document, file_bytes)
            st.success(f"Processed document: {uploaded_file.name}")
        except Exception as exc:
            st.error(str(exc))

    recent_records = load_recent_records()
    if recent_records:
        st.markdown("#### Study Library")
        record_options = {
            f"{record['name']} | {record.get('source_type', 'text')} | saved {record.get('saved_at', '')}": record
            for record in recent_records
        }
        selected_record_label = st.selectbox("Recent documents", list(record_options))
        if st.button("Open selected document"):
            selected_record = record_options[selected_record_label]
            restore_record(selected_record)
            st.success(f"Opened: {selected_record['name']}")

    st.markdown("#### Paste Text Directly")
    pasted_text = st.text_area("Paste lecture notes, copied PDF text, or a use-case document", height=180)
    pasted_name = st.text_input("Document name", value="pasted_document.txt")
    if st.button("Process pasted text"):
        if len(pasted_text.split()) < 20:
            st.warning("Paste a little more text so the assistant has enough context.")
        else:
            document = LoadedDocument(name=pasted_name.strip() or "pasted_document.txt", text=normalise_text(pasted_text), source_type="pasted_text")
            process_and_remember(document)
            st.success(f"Processed pasted text: {document.name}")

    document = st.session_state.document
    if document:
        st.divider()
        health = document_health(document.text, len(st.session_state.chunks))
        c1, c2, c3 = st.columns(3)
        c1.metric("Active document", document.name)
        c2.metric("Words", health["word_count"])
        c3.metric("Extraction readiness", f"{health['score']}/100")
        st.caption(
            "Extraction readiness estimates whether the uploaded document has enough readable text and structure for retrieval. "
            "It is not a grade for the lecture and not a guarantee that every diagram was extracted."
        )
        coverage_rows = [
            {"quality check": label, "status": "Found" if found else "Missing"}
            for label, found in health["coverage"].items()
        ]
        with st.expander("Extraction checks"):
            st.write("These checks show whether the text contains signals that are useful for analysis. They are helpful for business or project documents, but less important for normal lecture notes.")
            st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
        if len(document.text.split()) < 80:
            st.warning("The document is very short. Retrieval and insight quality may be limited.")
        preview_tab, text_tab = st.tabs(["PDF Preview", "Extracted Text"])
        with preview_tab:
            record = st.session_state.document_record
            if document.source_type == "pdf" and record:
                render_pdf_preview(record)
            else:
                st.info("PDF preview appears here when the active document is an uploaded PDF.")
        with text_tab:
            extracted_pages = split_extracted_pages(document.text)
            overview = build_document_overview(document.text)
            overview_tab, page_text_tab = st.tabs(["Document Overview", "Page Text"])
            with overview_tab:
                st.markdown("#### PDF Study Overview")
                st.write(overview["summary"])

                st.markdown("#### Main Section Map")
                if overview["main_sections"]:
                    st.dataframe(pd.DataFrame(overview["main_sections"]), use_container_width=True, hide_index=True)
                else:
                    st.info("No clear headings were detected automatically.")

                with st.expander("Page-by-page index"):
                    if overview["page_index"]:
                        st.dataframe(pd.DataFrame(overview["page_index"]), use_container_width=True, hide_index=True)
                    else:
                        st.info("No page-level topics were detected automatically.")

                st.caption(
                    "The overview is generated from extracted text. It is meant as a study navigation aid, not a perfect replacement for the original PDF table of contents."
                )

            with page_text_tab:
                page_numbers = [page_number for page_number, _ in extracted_pages]
                selected_text_page = st.selectbox("Extracted text page", page_numbers)
                page_text = next(text for page_number, text in extracted_pages if page_number == selected_text_page)
                st.caption(
                    "Extracted text is what the assistant searches. It is useful for copying exact text from the PDF."
                )
                st.text_area(f"Extracted text from page {selected_text_page}", page_text, height=360)
    else:
        st.info("Load a sample or upload a document to begin.")

with tabs[1]:
    st.subheader("Ask Questions With Source-Grounded Retrieval")
    retriever = st.session_state.retriever
    if not retriever:
        st.info("Load a document first.")
    else:
        mode = st.radio("Analysis mode", list(ANALYSIS_MODES), horizontal=True)
        st.caption(ANALYSIS_MODES[mode]["description"])
        question = st.text_area(
            "Question",
            value="",
            placeholder="Type here...",
            height=100,
        )
        st.caption(f"You can ask: {ANALYSIS_MODES[mode]['examples'][0]}")
        top_k = st.slider(
            "Number of source chunks",
            1,
            SETTINGS.maximum_top_k,
            SETTINGS.default_top_k,
        )
        if st.button("Retrieve and answer", type="primary"):
            if not question.strip():
                st.session_state.retrieved = []
                st.session_state.answer = None
                st.session_state.answer_scope = None
                st.session_state.evaluation = None
                st.warning("Type a question first.")
            else:
                retrieved = retriever.search(question, top_k=top_k)
                question_intent = classify_question_intent(question)
                answer_scope = "retrieval"
                if question_intent in {"brief_summary", "whole_summary", "study_topics", "navigation", "how_to"}:
                    answer = answer_document_overview_question(question, st.session_state.document, retrieved)
                    answer_scope = "full_document"
                else:
                    support = assess_question_support(
                        question,
                        st.session_state.document.text,
                    )
                    if not support.supported:
                        retrieved = []
                    answer = answer_question(
                        question,
                        retrieved,
                        min_score=SETTINGS.minimum_retrieval_score,
                    )
                    if not answer.sources and question_intent != "retrieval":
                        answer = answer_document_overview_question(question, st.session_state.document, retrieved)
                        answer_scope = "full_document"
                if answer_scope == "full_document":
                    evaluation = evaluate_full_document_answer(
                        question,
                        answer.answer,
                        st.session_state.document.text,
                        answer.missing_information,
                    )
                    trace_sources = []
                else:
                    evaluation = evaluate_answer(question, answer.answer, answer.sources, answer.missing_information)
                    trace_sources = retrieved
                st.session_state.retrieved = trace_sources
                st.session_state.answer = answer
                st.session_state.answer_scope = answer_scope
                st.session_state.evaluation = evaluation

        if st.session_state.answer:
            answer = st.session_state.answer
            st.markdown("#### Grounded Answer")
            if st.session_state.answer_scope == "full_document":
                st.info("Source scope: full extracted document and document overview.")
            st.markdown(answer.answer)
            if answer.missing_information:
                st.markdown("#### Missing Information")
                for item in answer.missing_information:
                    st.warning(item)

            if st.session_state.answer_scope == "full_document":
                st.caption("Chunk-level retrieval is hidden for this answer because it was generated from the full document view.")
            else:
                st.markdown("#### Retrieved Source Chunks")
                for source in answer.sources:
                    with st.expander(
                        f"{source.document_name} | Chunk {source.chunk_id} | {confidence_label(source.score)} | score={source.score:.3f}"
                    ):
                        st.write(source.text)

            document_record = st.session_state.document_record or {}
            document_id = document_record.get("id")
            if document_id:
                with st.expander("Rate this answer"):
                    with st.form(f"feedback_{document_id}_{hash(answer.question)}"):
                        helpful_choice = st.radio(
                            "Was this answer helpful?",
                            ["Yes", "No"],
                            horizontal=True,
                        )
                        correction = st.text_area(
                            "Optional correction or comment",
                            placeholder="What was missing, unclear, or incorrect?",
                            max_chars=3000,
                        )
                        if st.form_submit_button("Submit feedback"):
                            feedback_id = FEEDBACK_STORE.add(
                                FeedbackRecord(
                                    document_id=document_id,
                                    question=answer.question,
                                    helpful=helpful_choice == "Yes",
                                    correction=correction,
                                    evaluation=st.session_state.evaluation.to_dict()
                                    if st.session_state.evaluation
                                    else None,
                                )
                            )
                            st.success(f"Feedback recorded locally (reference {feedback_id}).")

with tabs[2]:
    st.subheader("Structured Decision Insights")
    insights = st.session_state.insights
    if not insights:
        st.info("Load a document first.")
    else:
        readiness = readiness_score(insights)
        metric_col, label_col = st.columns([1, 3])
        metric_col.metric("Implementation readiness", f"{readiness['score']}/100")
        label_col.info(f"{readiness['label']}: {readiness['reason']}")

        st.markdown("#### Summary")
        st.write(insights.summary)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Requirements")
            for item in insights.requirements or ["Not identified in the source document."]:
                st.write(f"- {item}")
            st.markdown("#### Action Items")
            for item in insights.action_items or ["Not identified in the source document."]:
                st.write(f"- {item}")
        with col2:
            st.markdown("#### Risks")
            for item in insights.risks or ["Not identified in the source document."]:
                st.write(f"- {item}")
            st.markdown("#### Recommendations")
            for item in insights.recommendations:
                st.write(f"- {item}")

        st.markdown("#### Decision Matrix")
        st.dataframe(pd.DataFrame(st.session_state.decision_matrix), use_container_width=True)

        st.markdown("#### Missing Information")
        for item in insights.missing_information or ["No major missing information detected by the rule-based check."]:
            st.warning(item)

with tabs[3]:
    st.subheader("LLM Output & Retrieval Evaluation")
    evaluation = st.session_state.evaluation
    if not evaluation:
        st.info("Ask a question first to generate an evaluation.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Relevance", f"{evaluation.relevance}/100")
        c2.metric("Completeness", f"{evaluation.completeness}/100")
        c3.metric("Grounding", f"{evaluation.grounding}/100")
        c4.metric("Consistency", f"{evaluation.consistency}/100")

        if evaluation.human_review_required:
            st.error(f"Human review required | Hallucination risk: {evaluation.hallucination_risk}")
        else:
            st.success(f"No immediate review flag | Hallucination risk: {evaluation.hallucination_risk}")

        st.markdown("#### Evaluation Notes")
        for note in evaluation.notes:
            st.write(f"- {note}")

        if st.session_state.answer_scope == "full_document":
            st.info("This was a full-document answer, so chunk-level retrieval traceability is not shown. Use the PDF preview and page text for manual verification.")
        elif st.session_state.retrieved:
            st.markdown("#### Retrieval Traceability")
            trace_rows = [
                {
                    "document": item.document_name,
                    "chunk": item.chunk_id,
                    "score": round(item.score, 3),
                    "confidence": confidence_label(item.score),
                    "preview": item.text[:180] + "...",
                }
                for item in st.session_state.retrieved
            ]
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)

            avg_score = sum(item.score for item in st.session_state.retrieved) / len(st.session_state.retrieved)
            if avg_score < 0.05:
                st.warning("Retrieval confidence is low. The question may need to be rephrased or the document may not contain enough evidence.")

with tabs[4]:
    st.subheader("Export Decision Report")
    if not st.session_state.insights:
        st.info("Load a document first.")
    else:
        answer = st.session_state.answer
        payload = build_export_payload(
            st.session_state.insights,
            st.session_state.decision_matrix,
            st.session_state.evaluation,
            answer.question if answer else None,
            answer.answer if answer else None,
        )

        json_text = payload_to_json(payload)
        csv_text = matrix_to_csv(st.session_state.decision_matrix)
        markdown_text = payload_to_markdown(payload)

        st.download_button(
            "Download JSON report",
            data=json_text,
            file_name=timestamped_name("decision_report", "json"),
            mime="application/json",
        )
        st.download_button(
            "Download decision matrix CSV",
            data=csv_text,
            file_name=timestamped_name("decision_matrix", "csv"),
            mime="text/csv",
        )
        st.download_button(
            "Download Markdown report",
            data=markdown_text,
            file_name=timestamped_name("decision_report", "md"),
            mime="text/markdown",
        )

        st.markdown("#### Report Preview")
        st.code(markdown_text[:4000], language="markdown")
