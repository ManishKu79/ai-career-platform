
import requests

# typing: type annotations
from typing import Dict, Any, Optional, List, Tuple

# io: in-memory file handling
import io

# logging
import logging

# os: for environment variable access
import os

logger = logging.getLogger(__name__)

# Base URL for FastAPI backend
# Reads from environment variable with fallback to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1       = f"{API_BASE_URL}/api/v1"

# Default timeout for all requests (seconds)
DEFAULT_TIMEOUT = 30


class APIClient:
    """
    HTTP client for the AI Career Platform FastAPI backend.

    All methods return either:
    - Success: {"success": True, "data": <response_data>}
    - Error:   {"success": False, "error": "<message>", "status_code": N}

    This consistent structure lets page code use:
        result = client.upload_resume(file_bytes, filename)
        if result["success"]:
            file_id = result["data"]["file_id"]
        else:
            st.error(result["error"])
    """

    def __init__(self, base_url: str = API_V1, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout  = timeout

        # Session reuses TCP connections across requests
        # More efficient than creating a new connection per call
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Executes a GET request and returns standardized result.

        Args:
            endpoint: API path (e.g., "/jobs/")
            params:   Query parameters dict

        Returns:
            Standardized result dict with success flag
        """
        try:
            url      = f"{self.base_url}{endpoint}"
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            return self._handle_response(response)

        except requests.ConnectionError:
            return {
                "success": False,
                "error":   f"Cannot connect to API at {self.base_url}. "
                           "Is the FastAPI server running?",
                "status_code": 0,
            }
        except requests.Timeout:
            return {
                "success":    False,
                "error":      f"Request timed out after {self.timeout}s",
                "status_code": 408,
            }
        except Exception as e:
            logger.error(f"GET {endpoint} failed: {e}")
            return {"success": False, "error": str(e), "status_code": 0}

    def _post(
        self,
        endpoint: str,
        json: Optional[Dict]  = None,
        data: Optional[Dict]  = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Executes a POST request and returns standardized result.

        Args:
            endpoint: API path
            json:     JSON body (Content-Type: application/json)
            data:     Form data (Content-Type: multipart/form-data)
            files:    File uploads (part of multipart)
            params:   Query parameters

        Returns:
            Standardized result dict
        """
        try:
            url      = f"{self.base_url}{endpoint}"
            response = self.session.post(
                url,
                json=json,
                data=data,
                files=files,
                params=params,
                timeout=self.timeout
            )
            return self._handle_response(response)

        except requests.ConnectionError:
            return {
                "success": False,
                "error":   f"Cannot connect to API at {self.base_url}.",
                "status_code": 0,
            }
        except requests.Timeout:
            return {
                "success":    False,
                "error":      f"Request timed out after {self.timeout}s",
                "status_code": 408,
            }
        except Exception as e:
            logger.error(f"POST {endpoint} failed: {e}")
            return {"success": False, "error": str(e), "status_code": 0}

    def _handle_response(self, response: requests.Response) -> Dict:
        """
        Converts requests.Response into standardized result dict.

        Handles:
        - 2xx: success → return parsed JSON as data
        - 4xx: client error → return error message
        - 5xx: server error → return error message
        - Non-JSON response → return raw text as error

        Args:
            response: requests.Response object

        Returns:
            Standardized result dict
        """
        try:
            body = response.json()
        except Exception:
            # Response was not JSON (e.g., HTML error page)
            body = {"detail": response.text[:200]}

        if response.ok:  # status 200-299
            return {"success": True, "data": body}
        else:
            # Extract error detail from response body
            detail = (
                body.get("detail") or
                body.get("error")  or
                f"HTTP {response.status_code}"
            )
            return {
                "success":    False,
                "error":      str(detail),
                "status_code": response.status_code,
            }

    # ─────────────────────────────────────────────────────────────────
    # PIPELINE ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def process_pipeline(
        self,
        file_bytes: bytes,
        filename: str,
        job_id: str,
        run_ranking: bool = True
    ) -> Dict:
        """
        Calls POST /pipeline/process — full 5-stage pipeline.

        Sends file + job_id as multipart/form-data.
        This is the primary endpoint called by the Upload page.

        Args:
            file_bytes:  Raw bytes of the uploaded resume file
            filename:    Original filename with extension
            job_id:      Job ID to score against
            run_ranking: Whether to trigger background re-ranking

        Returns:
            Full pipeline result with parse, NLP, skills, and ATS score
        """
        return self._post(
            "/pipeline/process",
            files={"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            data={"job_id": job_id, "run_ranking": str(run_ranking).lower()},
        )

    def batch_process(
        self,
        files: List[Tuple[str, bytes]],
        job_id: str
    ) -> Dict:
        """
        Calls POST /pipeline/process/batch — multiple files.

        Args:
            files:  List of (filename, file_bytes) tuples
            job_id: Job ID to score all resumes against

        Returns:
            Batch processing summary with per-file results
        """
        multipart_files = [
            ("files", (fname, io.BytesIO(fbytes), "application/octet-stream"))
            for fname, fbytes in files
        ]
        return self._post(
            "/pipeline/process/batch",
            files=multipart_files,
            data={"job_id": job_id},
        )

    def get_pipeline_status(self, file_id: str) -> Dict:
        """Gets current pipeline status for a resume."""
        return self._get(f"/pipeline/status/{file_id}")

    # ─────────────────────────────────────────────────────────────────
    # JOB ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def list_jobs(self, active_only: bool = True) -> Dict:
        """Lists all job descriptions."""
        return self._get("/jobs/", params={"active_only": active_only})

    def create_job(self, job_data: Dict) -> Dict:
        """Creates a new job description."""
        return self._post("/jobs/", json=job_data)

    def get_job(self, job_id: str) -> Dict:
        """Fetches a specific job by ID."""
        return self._get(f"/jobs/{job_id}")

    def deactivate_job(self, job_id: str) -> Dict:
        """Soft-deletes a job (sets is_active=False)."""
        try:
            url      = f"{self.base_url}/jobs/{job_id}"
            response = self.session.delete(url, timeout=self.timeout)
            return self._handle_response(response)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # RESUME ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def list_resumes(self, limit: int = 50, skip: int = 0) -> Dict:
        """Lists uploaded resumes with pagination."""
        return self._get("/upload/resumes", params={"limit": limit, "skip": skip})

    def get_resume(self, file_id: str) -> Dict:
        """Fetches a specific resume by file_id."""
        return self._get(f"/upload/resume/{file_id}")

    # ─────────────────────────────────────────────────────────────────
    # SCORING ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def score_resume(self, resume_file_id: str, job_id: str) -> Dict:
        """Scores a single resume against a job."""
        return self._post(
            "/scoring/score",
            params={"resume_file_id": resume_file_id, "job_id": job_id}
        )

    def batch_score(self, job_id: str, threshold: float = 0.0) -> Dict:
        """Batch scores all resumes against a job."""
        return self._post(
            "/scoring/batch",
            json={"job_id": job_id, "score_threshold": threshold}
        )

    def get_leaderboard(
        self,
        job_id: str,
        limit: int = 50,
        min_score: float = 0.0
    ) -> Dict:
        """Returns ranked candidates for a job."""
        return self._get(
            f"/scoring/leaderboard/{job_id}",
            params={"limit": limit, "min_score": min_score}
        )

    # ─────────────────────────────────────────────────────────────────
    # RANKING ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def rank_candidates(self, job_id: str) -> Dict:
        """Generates/refreshes candidate ranking for a job."""
        return self._post(f"/candidates/rank/{job_id}")

    def get_ranking(self, job_id: str) -> Dict:
        """Fetches most recent ranking for a job."""
        return self._get(f"/candidates/ranking/{job_id}")

    def get_shortlist(self, job_id: str, tier: str = "priority") -> Dict:
        """Returns shortlisted candidates by tier."""
        return self._get(
            f"/candidates/shortlist/{job_id}",
            params={"tier": tier}
        )

    def get_pool_statistics(self, job_id: str) -> Dict:
        """Returns score pool statistics for a job."""
        return self._get(f"/candidates/statistics/{job_id}")

    # ─────────────────────────────────────────────────────────────────
    # ADMIN / ANALYTICS ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def get_health(self) -> Dict:
        """Database health check."""
        return self._get("/admin/health")

    def get_db_stats(self) -> Dict:
        """Database statistics."""
        return self._get("/admin/stats")

    def get_top_skills(self, limit: int = 20) -> Dict:
        """Top skills across all resumes."""
        return self._get("/admin/analytics/top-skills", params={"limit": limit})

    def get_score_distribution(self, job_id: str) -> Dict:
        """Score histogram for a job."""
        return self._get(f"/admin/analytics/score-distribution/{job_id}")

    def get_skill_gap_heatmap(self, job_id: str) -> Dict:
        """Skill gap analysis for a job."""
        return self._get(f"/admin/analytics/skill-gap/{job_id}")

    def get_hiring_funnel(self, job_id: Optional[str] = None) -> Dict:
        """Hiring funnel metrics."""
        params = {"job_id": job_id} if job_id else {}
        return self._get("/admin/analytics/funnel", params=params)

    def get_pipeline_summary(self) -> Dict:
        """Resume processing pipeline status summary."""
        return self._get("/admin/pipeline/status")

    def check_api_health(self) -> bool:
        """
        Quick connectivity check.
        Returns True if FastAPI is reachable, False otherwise.
        Used by app.py to display connection status in sidebar.
        """
        try:
            response = self.session.get(
                f"{API_BASE_URL}/health",
                timeout=3
            )
            return response.ok
        except Exception:
            return False


# Module-level singleton — imported by all page files
client = APIClient()
