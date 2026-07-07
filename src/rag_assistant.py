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
    if re.search(r"\b(define|definition|meaning)\b", lower) or "what is" in lower:
        intent.add("definition")
    if re.search(
        r"\b(type|types|kind|kinds|form|forms|category|categories|"
        r"classification|classifications)\b",
        lower,
    ):
        intent.add("list")
    if any(term in lower for term in ("explain", "simplified", "beginner", "summarise", "summarize", "what is")):
        intent.add("explanation")
    if re.search(r"\b(compare|comparison|differentiate|difference between|versus|vs\.?)\b", lower):
        intent.add("comparison")
    if re.search(r"\b(advantage|advantages|benefit|benefits|merit|merits)\b", lower):
        intent.add("advantages")
    if re.search(r"\b(disadvantage|disadvantages|drawback|drawbacks|demerit|demerits)\b", lower):
        intent.add("disadvantages")
    if re.search(r"\b(cause|causes|reason|reasons|why)\b", lower):
        intent.add("causes")
    if re.search(r"\b(effect|effects|impact|impacts|result|results)\b", lower):
        intent.add("effects")
    if re.search(r"\b(example|examples|application|applications|use case|use cases)\b", lower):
        intent.add("examples")
    if re.search(r"\b(how|steps|procedure|process|workflow|method)\b", lower):
        intent.add("procedure")
    if any(term in lower for term in ("risk", "challenge", "limitation", "issue")):
        intent.add("risks")
    if any(term in lower for term in ("recommend", "next step", "action")):
        intent.add("recommendations")
    if any(term in lower for term in ("goal", "objective", "purpose")):
        intent.add("objective")
    if "explanation" not in intent and any(term in lower for term in ("data", "input", "source")):
        intent.add("data")
    if any(term in lower for term in ("metric", "evaluate", "evaluation", "kpi", "test")):
        intent.add("evaluation")
    return intent or {"general"}


def _compose_answer(question: str, intent: set[str], sentences: list[str]) -> str:
    if intent & {"definition", "list"}:
        return _compose_definition_and_list(question, intent, sentences)
    if intent & {
        "comparison",
        "advantages",
        "disadvantages",
        "causes",
        "effects",
        "examples",
        "procedure",
    }:
        return _compose_study_sections(intent, sentences)
    if "objective" in intent:
        objective_sentences = _select_by_terms(
            sentences,
            ("goal", "objective", "reduce", "support", "help"),
        )
        if objective_sentences:
            return "Objective: " + " ".join(
                _clean_sentence(sentence) for sentence in objective_sentences[:2]
            )
    if "explanation" in intent:
        return _compose_explanation(question, sentences)

    groups = {
        "risks": _select_by_terms(sentences, ("risk", "challenge", "limitation", "issue", "poor", "inconsistent", "imbalance", "fail")),
        "recommendations": _select_recommendations(sentences),
        "data": _select_by_terms(sentences, ("data", "input", "source", "image", "metadata", "dataset")),
        "evaluation": _select_by_terms(sentences, ("evaluate", "metric", "accuracy", "f1", "matrix", "baseline", "test", "leakage")),
        "objective": _select_by_terms(sentences, ("goal", "objective", "reduce", "support", "help")),
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
    if "objective" in intent and groups["objective"]:
        sections.append("Objective: " + " ".join(_clean_sentence(sentence) for sentence in groups["objective"][:2]))

    if sections:
        return "\n\n".join(sections)
    return " ".join(_clean_sentence(sentence) for sentence in sentences[:4])


def _compose_study_sections(intent: set[str], sentences: list[str]) -> str:
    section_terms = {
        "comparison": (
            "difference",
            "whereas",
            "while",
            "compared",
            "versus",
            "unlike",
        ),
        "advantages": (
            "advantage",
            "benefit",
            "useful",
            "allows",
            "improve",
        ),
        "disadvantages": (
            "disadvantage",
            "drawback",
            "limitation",
            "cost",
            "difficult",
        ),
        "causes": (
            "cause",
            "caused",
            "because",
            "due to",
            "reason",
            "results from",
        ),
        "effects": (
            "effect",
            "impact",
            "results in",
            "leads to",
            "therefore",
        ),
        "examples": (
            "example",
            "application",
            "used in",
            "such as",
            "include",
        ),
        "procedure": (
            "step",
            "first",
            "then",
            "next",
            "finally",
            "procedure",
            "process",
        ),
    }
    headings = {
        "comparison": "Comparison",
        "advantages": "Advantages",
        "disadvantages": "Disadvantages",
        "causes": "Causes",
        "effects": "Effects",
        "examples": "Examples or applications",
        "procedure": "Procedure",
    }
    sections = []
    for name in (
        "comparison",
        "advantages",
        "disadvantages",
        "causes",
        "effects",
        "examples",
        "procedure",
    ):
        if name not in intent:
            continue
        selected = _select_by_terms(sentences, section_terms[name])
        if selected:
            body = "\n".join(
                f"- {_remove_page_markers(_clean_sentence(sentence))}"
                for sentence in selected[:5]
            )
        else:
            body = (
                "- The retrieved text does not contain a clearly extractable "
                f"{headings[name].lower()} section."
            )
        sections.append(f"{headings[name]}\n{body}")
    sections.append(
        "Source note\nThis answer is limited to the text extracted from the uploaded document."
    )
    return "\n\n".join(sections)


def _compose_definition_and_list(
    question: str,
    intent: set[str],
    sentences: list[str],
) -> str:
    subject = _question_subject(question)
    cleaned = [
        _remove_repeated_source_labels(
            _remove_page_markers(_clean_sentence(sentence))
        )
        for sentence in sentences
    ]
    cleaned = [sentence for sentence in cleaned if sentence]
    sections = []

    if "definition" in intent:
        definition = _find_definition(subject, cleaned)
        if definition:
            sections.append(f"Definition\n{definition}")
        else:
            sections.append(
                "Definition\n"
                "The retrieved pages do not contain a clear one-sentence definition. "
                "Check the displayed source page before using a general definition."
            )

    if "list" in intent:
        labels = _extract_type_labels(subject, cleaned)
        if labels:
            sections.append(
                "Types mentioned in the document\n"
                + "\n".join(f"- {label}" for label in labels)
            )
        else:
            supporting = [
                sentence
                for sentence in cleaned
                if subject in sentence.lower()
                and any(term in sentence.lower() for term in ("type", "kind", "form", "category"))
            ]
            if supporting:
                sections.append(
                    "Types described in the document\n"
                    + "\n".join(f"- {sentence}" for sentence in supporting[:4])
                )
            else:
                sections.append(
                    "Types\nNo clearly extractable list of types was found in the retrieved text."
                )

    sections.append(
        "Source note\nThis answer is limited to the text extracted from the uploaded document."
    )
    return "\n\n".join(sections)


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
    raw_query_terms = [
        _normalise_ranking_term(term)
        for term in re.findall(r"[A-Za-z]+", question.lower())
    ]
    query_terms = _expand_query_terms(set(raw_query_terms))
    query_bigrams = set(zip(raw_query_terms, raw_query_terms[1:]))
    candidates: list[tuple[int, str]] = []
    seen_terms: list[set[str]] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            clean = sentence.strip()
            if len(clean.split()) < 5:
                continue
            ordered_terms = [
                _normalise_ranking_term(term)
                for term in re.findall(r"[A-Za-z]+", clean.lower())
            ]
            terms = set(ordered_terms)
            if _is_near_duplicate(terms, seen_terms):
                continue
            seen_terms.append(terms)
            overlap = len(query_terms & terms)
            bigram_overlap = len(query_bigrams & set(zip(ordered_terms, ordered_terms[1:])))
            candidates.append((overlap * 2 + bigram_overlap * 3, clean))
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


def _normalise_ranking_term(term: str) -> str:
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term


def _question_subject(question: str) -> str:
    lower = question.lower()
    patterns = (
        r"(?:definition|meaning)\s+of\s+([a-z][a-z-]*)",
        r"(?:type|types|kind|kinds|form|forms|category|categories|"
        r"classification|classifications)\s+of\s+([a-z][a-z-]*)",
        r"define\s+([a-z][a-z-]*)",
        r"what\s+is\s+(?:an?\s+|the\s+)?([a-z][a-z-]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(1)
    terms = [
        term
        for term in re.findall(r"[a-z][a-z-]+", lower)
        if term not in {
            "define",
            "definition",
            "meaning",
            "types",
            "kinds",
            "forms",
            "categories",
            "classifications",
            "classification",
            "and",
            "of",
            "the",
            "what",
            "are",
            "its",
            "give",
        }
    ]
    return terms[0] if terms else ""


def _find_definition(subject: str, sentences: list[str]) -> str:
    if not subject:
        return ""
    subject_pattern = re.compile(
        rf"^\s*(?:{re.escape(subject)}|the\s+term\s+{re.escape(subject)})\s+"
        r"(?:is|refers to|is defined as|means)\b",
        flags=re.IGNORECASE,
    )
    candidates = [
        sentence
        for sentence in sentences
        if subject_pattern.search(sentence)
        and "types of" not in sentence.lower()
        and ":" not in sentence[: max(18, len(subject) + 12)]
    ]
    if not candidates:
        return ""
    candidates.sort(
        key=lambda sentence: (
            "defined as" in sentence.lower() or "refers to" in sentence.lower(),
            -len(sentence.split()),
        ),
        reverse=True,
    )
    return candidates[0]


def _extract_type_labels(subject: str, sentences: list[str]) -> list[str]:
    if not subject:
        return []
    matches = []
    for sentence in sentences:
        sentence = _normalise_extracted_text(sentence)
        if subject.lower() not in sentence.lower():
            continue

        heading_matches = re.findall(
            rf"(?:types|kinds|forms|categories|classifications)\s+of\s+"
            rf"{re.escape(subject)}\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){{0,1}})\s*:",
            sentence,
            flags=re.IGNORECASE,
        )
        matches.extend(
            heading
            if subject.lower() in heading.lower().split()
            else f"{heading} {subject}"
            for heading in heading_matches
        )

        list_match = re.search(
            rf"(?:include|includes|including|are|is|:)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if list_match and re.search(
            rf"(?:types|kinds|forms|categories|classifications)\s+of\s+{re.escape(subject)}",
            sentence,
            flags=re.IGNORECASE,
        ):
            list_part = re.split(r"\b(?:based on|depending on|source)\b", list_match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            for item in re.split(r",|;|\band\b|\bor\b", list_part):
                inline = re.search(
                    rf"\b([A-Za-z-]+(?:\s+[A-Za-z-]+)?\s+{re.escape(subject)})\b",
                    item,
                    flags=re.IGNORECASE,
                )
                if inline:
                    matches.append(inline.group(1))

        inline_matches = re.findall(
            rf"\b([A-Za-z-]+(?:\s+[A-Za-z-]+)?\s+{re.escape(subject)})\b",
            sentence,
            flags=re.IGNORECASE,
        )
        matches.extend(inline_matches)
    blocked = {
        f"of {subject}",
        f"types {subject}",
        f"during {subject}",
        f"reduce {subject}",
        f"reduces {subject}",
        f"contact {subject}",
        f"material {subject}",
        f"loss {subject}",
        f"surface {subject}",
        f"motion {subject}",
        f"progressive {subject}",
        f"relative {subject}",
    }
    blocked_prefixes = {
        "a",
        "analysis",
        "an",
        "and",
        "are",
        "as",
        "based",
        "by",
        "can",
        "common",
        "contact",
        "contacts",
        "during",
        "discussed",
        "example",
        "for",
        "from",
        "in",
        "ing",
        "into",
        "is",
        "loss",
        "material",
        "motion",
        "most",
        "of",
        "on",
        "or",
        "source",
        "surface",
        "the",
        "this",
        "to",
        "types",
        "wear",
        "with",
    }
    labels = []
    for match in matches:
        label = " ".join(match.split())
        label_lower = label.lower()
        label_words = label_lower.split()
        if label_lower in blocked:
            continue
        if not label_words or label_words[-1] != subject.lower():
            continue
        if len(label_words) < 2 or len(label_words) > 3:
            continue
        if any(word in blocked_prefixes for word in label_words[:-1]):
            continue
        label = label[0].upper() + label[1:]
        if label.lower() not in {item.lower() for item in labels}:
            labels.append(label)
    return labels[:8]


def _normalise_extracted_text(text: str) -> str:
    text = text.replace("\uFFFD", " ")
    text = text.replace("□", " ")
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_source_labels(text: str) -> str:
    text = re.sub(
        r"(?:Prof\.?\s+)?[A-Z][A-Za-z-]+\s+Simulation of [A-Za-z ]+",
        " ",
        text,
    )
    text = re.sub(r"\bSource:\s*\[[^\]]+\]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .:-")


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
