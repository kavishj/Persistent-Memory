from core.writer.extractor import extract_facts, ExtractedFact
from core.writer.classifier import classify_session, ClassificationResult
from core.writer.deduplicator import check_duplicate, DedupAction, DedupResult
from core.writer.conflict_resolver import resolve_conflict, ConflictResult, ConflictResolution