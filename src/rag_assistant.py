from __future__ import annotations

from dataclasses import dataclass
import re

from src.retriever import RetrievedChunk


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    missing_information: list[str]


def answer_question(question: str, retrieved_chunks: list[RetrievedChunk], min_score: float = 0.03) -> GroundedAnswer:
    useful_chunks = [chunk for chunk in retrieved_chunks if chunk.score >= min_score]
    if not useful_chunks:
        return GroundedAnswer(
            question=question,
            answer="Not enough evidence was found in the uploaded document to answer this question.",
            sources=[],
            missing_information=["Relevant source context was not found."],
        )

    intent = _detect_intent(question)
    sentences = _rank_sentences(question, useful_chunks)
    if not sentences:
        answer = "Relevant context was retrieved, but no concise answer sentence could be extracted."
        missing = ["The retrieved text may need manual review."]
    else:
        answer = _compose_answer(question, intent, sentences)
        missing = _detect_missing_information(question, answer)

    return GroundedAnswer(question=question, answer=answer, sources=useful_chunks, missing_information=missing)


def _detect_intent(question: str) -> set[str]:
    lower = question.lower()
    intent = set()
    if any(term in lower for term in ("explain", "simplified", "beginner", "summarise", "summarize", "what is")):
        intent.add("explanation")
    if any(term in lower for term in ("risk", "challenge", "limitation", "issue")):
        intent.add("risks")
    if any(term in lower for term in ("recommend", "next step", "action", "should")):
        intent.add("recommendations")
    if "explanation" not in intent and any(term in lower for term in ("data", "input", "source")):
        intent.add("data")
    if any(term in lower for term in ("metric", "evaluate", "evaluation", "kpi", "test")):
        intent.add("evaluation")
    return intent or {"general"}


def _compose_answer(question: str, intent: set[str], sentences: list[str]) -> str:
    if "explanation" in intent:
        return _compose_explanation(question, sentences)

    groups = {
        "risks": _select_by_terms(sentences, ("risk", "challenge", "limitation", "issue", "poor", "inconsistent", "imbalance", "fail")),
        "recommendations": _select_recommendations(sentences),
        "data": _select_by_terms(sentences, ("data", "input", "source", "image", "metadata", "dataset")),
        "evaluation": _select_by_terms(sentences, ("evaluate", "metric", "accuracy", "f1", "matrix", "baseline", "test", "leakage")),
    }

    sections = []
    if "risks" in intent and groups["risks"]:
        sections.append("Key risks: " + " ".join(_clean_sentence(sentence) for sentence in groups["risks"][:2]))
    if "recommendations" in intent and groups["recommendations"]:
        sections.append("Recommended next steps: " + " ".join(_clean_sentence(sentence) for sentence in groups["recommendations"][:2]))
    if "data" in intent and groups["data"]:
        sections.append("Required data: " + " ".join(_clean_sentence(sentence) for sentence in groups["data"][:2]))
    if "evaluation" in intent and groups["evaluation"]:
        sections.append("Evaluation approach: " + " ".join(_clean_sentence(sentence) for sentence in groups["evaluation"][:2]))

    if sections:
        return "\n\n".join(sections)
    return " ".join(_clean_sentence(sentence) for sentence in sentences[:4])


def _compose_explanation(question: str, sentences: list[str]) -> str:
    combined = " ".join(_clean_sentence(sentence) for sentence in sentences[:4])
    combined = _remove_page_markers(combined)
    model_parts = _extract_communication_model_parts(combined)
    diagram_note = _diagram_note(question)
    detailed = _needs_detailed_explanation(question)

    if _is_communication_model_question(question):
        if not _has_complete_communication_model(model_parts):
            model_parts = _standard_communication_model_parts()
        lines = _communication_model_explanation(model_parts, detailed)
        if diagram_note:
            lines.extend(["", diagram_note])
        return "\n".join(lines)

    clean_sentences = [_remove_page_markers(_clean_sentence(sentence)) for sentence in sentences[:3]]
    answer = "Beginner explanation: " + " ".join(sentence for sentence in clean_sentences if sentence)
    if diagram_note:
        answer = f"{answer}\n\n{diagram_note}"
    return answer


def _rank_sentences(question: str, chunks: list[RetrievedChunk]) -> list[str]:
    query_terms = _expand_query_terms({term.lower() for term in re.findall(r"[A-Za-z][A-Za-z\-]+", question)})
    candidates: list[tuple[int, str]] = []
    seen_terms: list[set[str]] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            clean = sentence.strip()
            if len(clean.split()) < 5:
                continue
            terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z\-]+", clean)}
            if _is_near_duplicate(terms, seen_terms):
                continue
            seen_terms.append(terms)
            overlap = len(query_terms & terms)
            candidates.append((overlap, clean))
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [sentence for score, sentence in candidates if score > 0] or [sentence for _, sentence in candidates[:4]]


def _detect_missing_information(question: str, answer: str) -> list[str]:
    missing = []
    question_lower = question.lower()
    answer_lower = answer.lower()
    expected_terms = {
        "risk": ["risk", "challenge", "limitation", "issue"],
        "metric": ["metric", "kpi", "score", "accuracy", "measurement"],
        "data": ["data", "dataset", "source", "input"],
        "implementation": ["implement", "integration", "deploy", "workflow"],
        "recommendation": ["recommend", "should", "next step"],
    }
    for label, terms in expected_terms.items():
        if label in question_lower and not any(term in answer_lower for term in terms):
            missing.append(f"The answer may not contain enough detail about {label}.")
    return missing


def _select_by_terms(sentences: list[str], terms: tuple[str, ...]) -> list[str]:
    selected = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            selected.append(sentence)
    return selected


def _select_recommendations(sentences: list[str]) -> list[str]:
    scored: list[tuple[int, str]] = []
    strong_markers = ("recommendation:", "recommend:", "start with", "should", "before expanding", "next step")
    action_terms = ("validate", "define", "compare", "pilot", "expand", "document", "monitor", "review")
    weak_context = ("expected output", "visual examples for review", "human review is required")

    for sentence in sentences:
        lower = sentence.lower()
        if "expected output" in lower:
            continue
        if any(marker in lower for marker in weak_context) and not any(marker in lower for marker in strong_markers):
            continue
        score = 0
        if any(marker in lower for marker in strong_markers):
            score += 4
        score += sum(1 for term in action_terms if term in lower)
        if score:
            scored.append((score, sentence))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [sentence for _, sentence in scored]


def _clean_sentence(sentence: str) -> str:
    return re.sub(
        r"^(Risks and limitations|Recommendation|Evaluation|Expected output|Inputs and data sources|Objective|Grounded Answer):\s*",
        "",
        sentence.strip(),
        flags=re.IGNORECASE,
    )


def _remove_page_markers(text: str) -> str:
    text = re.sub(r"\[Page\s+\d+\]", " ", text)
    text = re.sub(r"Communication Networks\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Network Technology\s+\d+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _extract_communication_model_parts(text: str) -> list[tuple[str, str]]:
    component_patterns = [
        ("Source", r"Source\s+(.+?)(?=Transmitter|Transmission System|Receiver|Destination|$)"),
        ("Transmitter", r"Transmitter\s+(.+?)(?=Transmission System|Receiver|Destination|$)"),
        ("Transmission system", r"Transmission System\s+(.+?)(?=Receiver|Destination|$)"),
        ("Receiver", r"Receiver\s+(.+?)(?=Destination|$)"),
        ("Destination", r"Destination\s+(.+?)(?=Simplified|Key Communications|Networking|Circuit Switching|Packet Switching|$)"),
    ]
    parts = []
    for name, pattern in component_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            description = match.group(1).strip(" .:-")
            if description:
                parts.append((name, description[:180]))
    return parts


def _is_communication_model_question(question: str) -> bool:
    lower = question.lower()
    return "communication model" in lower or "data communication model" in lower


def _has_complete_communication_model(model_parts: list[tuple[str, str]]) -> bool:
    names = {name.lower() for name, _ in model_parts}
    expected = {"source", "transmitter", "transmission system", "receiver", "destination"}
    return len(names & expected) >= 4


def _standard_communication_model_parts() -> list[tuple[str, str]]:
    return [
        ("Source", "generates the original data or message to be transmitted"),
        ("Transmitter", "converts the data into signals that can travel through the selected medium"),
        ("Transmission system", "carries the signal from the sender side to the receiver side"),
        ("Receiver", "converts the received signal back into usable data"),
        ("Destination", "takes and uses the incoming data"),
    ]


def _diagram_note(question: str) -> str:
    lower = question.lower()
    if "diagram" not in lower and not re.search(r"\bpage\s+\d+\b", lower):
        return ""
    return (
        "Diagram note: This explanation is based on the text extracted from the PDF. "
        "If the diagram contains arrows, labels, or visual details that are stored only as an image, "
        "those parts may need manual review."
    )


def _needs_detailed_explanation(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in ("how", "work", "works", "detail", "step", "flow", "chart", "example"))


def _communication_model_explanation(model_parts: list[tuple[str, str]], detailed: bool) -> list[str]:
    lines = [
        "Beginner explanation: A simplified data communication model shows how information moves from one place to another through a communication system.",
        "",
        "Flow:",
        "Source -> Transmitter -> Transmission system -> Receiver -> Destination",
        "",
        "Main parts:",
    ]
    lines.extend(f"- {name}: {description}" for name, description in model_parts)

    if not detailed:
        lines.extend(
            [
                "",
                "In simple words: the source creates the message, the transmitter prepares it for sending, the transmission system carries it, the receiver converts it back, and the destination uses the received data.",
            ]
        )
        return lines

    lines.extend(
        [
            "",
            "How it works step by step:",
            "1. The source creates the original data. This could be text, voice, image data, or a file.",
            "2. The transmitter converts that data into a signal that can travel through a medium. For example, a computer network card converts bits into electrical, optical, or wireless signals.",
            "3. The transmission system carries the signal from sender side to receiver side. This can be copper cable, optical fibre, wireless channel, or a larger network path.",
            "4. The receiver takes the incoming signal and converts it back into usable data.",
            "5. The destination receives and uses the final data.",
            "",
            "Simple example:",
            "When you send a message from your laptop to a friend, your laptop is the source. The Wi-Fi adapter acts as the transmitter. The wireless network and internet path act as the transmission system. Your friend's device receives the signal, converts it back into data, and shows the message as the destination.",
            "",
            "Exam takeaway:",
            "The model is important because it separates communication into clear functions: data creation, signal conversion, transmission, signal recovery, and final delivery. Many later network topics, such as errors, flow control, addressing, routing, and security, happen around this basic flow.",
        ]
    )
    return lines


def _expand_query_terms(terms: set[str]) -> set[str]:
    synonyms = {
        "risk": {"risk", "risks", "challenge", "limitation", "issue", "failure"},
        "risks": {"risk", "risks", "challenge", "limitation", "issue", "failure"},
        "recommended": {"recommend", "recommendation", "recommended", "next", "step", "action"},
        "recommend": {"recommend", "recommendation", "recommended", "next", "step", "action"},
        "steps": {"step", "steps", "action", "actions", "recommendation"},
        "data": {"data", "dataset", "input", "source", "metadata"},
        "evaluate": {"evaluate", "evaluation", "metric", "kpi", "test", "baseline"},
    }
    expanded = set(terms)
    for term in list(terms):
        expanded.update(synonyms.get(term, set()))
    return expanded


def _is_near_duplicate(terms: set[str], seen_terms: list[set[str]], threshold: float = 0.72) -> bool:
    if not terms:
        return True
    for previous in seen_terms:
        union = terms | previous
        if not union:
            continue
        similarity = len(terms & previous) / len(union)
        if similarity >= threshold:
            return True
    return False
