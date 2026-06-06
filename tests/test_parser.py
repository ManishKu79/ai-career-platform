

import pytest
import io

# The service we are testing
from backend.services.parser import ResumeParser

# Our parser instance
parser = ResumeParser()


class TestTextCleaning:
    """Tests for the _clean_text method."""

    def test_removes_extra_whitespace(self):
        """Multiple spaces should become single spaces."""
        dirty = "Python   developer    with   experience"
        clean = parser._clean_text(dirty)
        assert "  " not in clean

    def test_removes_excess_newlines(self):
        """More than 2 consecutive newlines become 2."""
        dirty = "Section One\n\n\n\n\nSection Two"
        clean = parser._clean_text(dirty)
        assert "\n\n\n" not in clean

    def test_preserves_real_content(self):
        """Cleaning should not remove actual words."""
        text = "Python FastAPI MongoDB Docker"
        clean = parser._clean_text(text)
        assert "Python" in clean
        assert "FastAPI" in clean


class TestEmailExtraction:
    """Tests for the _extract_email method."""

    def test_extracts_standard_email(self):
        text = "Contact: john.doe@example.com for more info"
        assert parser._extract_email(text) == "john.doe@example.com"

    def test_extracts_email_uppercase_normalized(self):
        text = "Email: JOHN@COMPANY.COM"
        result = parser._extract_email(text)
        assert result == "john@company.com"

    def test_returns_none_when_no_email(self):
        text = "No email address in this text"
        assert parser._extract_email(text) is None

    def test_extracts_email_with_plus(self):
        """Gmail-style plus addressing should be captured."""
        text = "john+work@gmail.com"
        assert parser._extract_email(text) == "john+work@gmail.com"


class TestPhoneExtraction:
    """Tests for the _extract_phone method."""

    def test_extracts_formatted_phone(self):
        text = "Phone: (555) 123-4567"
        result = parser._extract_phone(text)
        assert result is not None
        assert "555" in result

    def test_extracts_dashed_phone(self):
        text = "Call me at 555-867-5309"
        result = parser._extract_phone(text)
        assert result is not None

    def test_returns_none_when_no_phone(self):
        text = "No phone number here"
        assert parser._extract_phone(text) is None


class TestFileValidation:
    """Tests for file type and size validation."""

    def test_rejects_unsupported_extension(self):
        """TXT files should be rejected."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(b"some content", "resume.txt")

    def test_rejects_oversized_file(self):
        """Files over 10MB should be rejected."""
        # Create fake bytes exceeding 10MB
        large_bytes = b"x" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds maximum"):
            parser.parse(large_bytes, "resume.pdf")

    def test_rejects_empty_text_extraction(self):
        """Files that yield no text should be rejected."""
        # Valid PDF extension but empty content that won't parse
        with pytest.raises((ValueError, Exception)):
            parser.parse(b"not a real pdf", "resume.pdf")
