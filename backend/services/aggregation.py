
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AggregationService:
    """
    MongoDB aggregation pipeline query library.

    Each method corresponds to one dashboard chart or metric.
    All methods return lists of dicts ready for:
    - Plotly chart construction (Module 9)
    - API response serialization
    - CSV export

    Aggregation Pipeline Reference:
    $match:   Filter documents (uses indexes when on indexed fields)
    $group:   Group and aggregate (SUM, AVG, COUNT, MIN, MAX)
    $sort:    Order results
    $project: Reshape documents (include/exclude/rename fields)
    $lookup:  Left outer join with another collection
    $unwind:  Deconstruct array field (one doc per element)
    $limit:   Limit output count
    $skip:    Skip N documents (pagination)
    $count:   Count documents into a single field
    $facet:   Multiple sub-pipelines in one stage
    """

    async def score_distribution_by_job(
        self,
        db: AsyncIOMotorDatabase,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Computes score distribution statistics for all candidates
        scored against a specific job.

        Aggregation pipeline:
        1. $match:   Filter to scores for this job_id
        2. $group:   Compute mean, min, max, std of final_score
                     Count total candidates
                     Bucket counts by score tier (using $cond)
        3. $project: Reshape output for chart consumption

        MongoDB $group operators used:
        - $avg:   Average of field values
        - $min:   Minimum value
        - $max:   Maximum value
        - $sum:   Sum (with 1 as argument = count)
        - $push:  Collect values into array (for histogram)

        Args:
            db:     Async MongoDB database
            job_id: Job to compute distribution for

        Returns:
            Dict with statistics and score buckets for histogram
        """
        pipeline = [
            # Stage 1: Filter to scores for this job
            # $match before $group is crucial for performance —
            # the idx_scores_job_id index makes this fast
            {"$match": {"job_id": job_id}},

            # Stage 2: Compute aggregate statistics
            # _id: null means group ALL matched docs into one group
            # (as opposed to grouping by a field value)
            {"$group": {
                "_id": None,  # Single group = statistics over all docs

                # $avg: arithmetic mean of final_score across all docs
                "mean_score":   {"$avg": "$final_score"},

                # $min/$max: extreme values
                "min_score":    {"$min": "$final_score"},
                "max_score":    {"$max": "$final_score"},

                # $sum with 1: counts total documents in group
                "total_count":  {"$sum": 1},

                # Conditional counting for tier buckets
                # $cond: { if: condition, then: value_if_true, else: value_if_false }
                # $sum of 1/0 = count of documents meeting condition
                "excellent_count": {
                    "$sum": {
                        "$cond": [{"$gte": ["$final_score", 0.80]}, 1, 0]
                    }
                },
                "good_count": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$gte": ["$final_score", 0.65]},
                                {"$lt":  ["$final_score", 0.80]}
                            ]}, 1, 0
                        ]
                    }
                },
                "fair_count": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$gte": ["$final_score", 0.50]},
                                {"$lt":  ["$final_score", 0.65]}
                            ]}, 1, 0
                        ]
                    }
                },
                "poor_count": {
                    "$sum": {
                        "$cond": [{"$lt": ["$final_score", 0.50]}, 1, 0]
                    }
                },

                # $push: collects all scores into an array
                # Used to compute histogram bins in Python
                # (MongoDB doesn't have a built-in histogram operator)
                "all_scores": {"$push": "$final_score"},
            }},

            # Stage 3: Reshape output
            # _id: 0 removes the grouping key from output
            {"$project": {
                "_id": 0,
                "mean_score":      {"$round": ["$mean_score", 4]},
                "min_score":       1,
                "max_score":       1,
                "total_count":     1,
                "excellent_count": 1,
                "good_count":      1,
                "fair_count":      1,
                "poor_count":      1,
                "all_scores":      1,
            }}
        ]

        results = await db["scores"].aggregate(pipeline).to_list(length=1)

        if not results:
            return {
                "job_id": job_id,
                "total_count": 0,
                "mean_score": 0,
                "tier_distribution": {},
                "histogram_data": []
            }

        data = results[0]

        # Build histogram bins in Python (10 bins from 0 to 1)
        all_scores = data.get("all_scores", [])
        histogram  = self._build_histogram(all_scores, bins=10)

        return {
            "job_id":        job_id,
            "total_count":   data.get("total_count", 0),
            "mean_score":    data.get("mean_score", 0),
            "min_score":     data.get("min_score", 0),
            "max_score":     data.get("max_score", 0),
            "tier_distribution": {
                "Excellent": data.get("excellent_count", 0),
                "Good":      data.get("good_count", 0),
                "Fair":      data.get("fair_count", 0),
                "Poor":      data.get("poor_count", 0),
            },
            "histogram_data": histogram,
        }

    def _build_histogram(
        self,
        scores: List[float],
        bins: int = 10
    ) -> List[Dict]:
        """
        Builds histogram bin data from a list of scores.

        Creates `bins` equal-width buckets from 0.0 to 1.0.
        Counts how many scores fall in each bucket.

        Example with bins=5:
        Buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        Count scores in each bucket → histogram bars

        Args:
            scores: List of float scores (0.0 to 1.0)
            bins:   Number of histogram buckets

        Returns:
            List of dicts: [{bin_start, bin_end, count, label}, ...]
        """
        if not scores:
            return []

        # Create equal-width bins
        bin_width = 1.0 / bins
        histogram  = []

        for i in range(bins):
            bin_start = round(i * bin_width, 2)
            bin_end   = round((i + 1) * bin_width, 2)

            # Count scores in [bin_start, bin_end)
            # Last bin includes bin_end (closed interval)
            if i == bins - 1:
                count = sum(1 for s in scores if bin_start <= s <= bin_end)
            else:
                count = sum(1 for s in scores if bin_start <= s < bin_end)

            histogram.append({
                "bin_start": bin_start,
                "bin_end":   bin_end,
                "count":     count,
                "label":     f"{int(bin_start*100)}-{int(bin_end*100)}%",
            })

        return histogram

    async def top_skills_across_resumes(
        self,
        db: AsyncIOMotorDatabase,
        limit: int = 20
    ) -> List[Dict]:
        """
        Finds the most common skills across all resumes in the database.

        Aggregation pipeline:
        1. $project: Extract extracted_skills array
        2. $unwind:  One document per skill (deconstruct array)
        3. $group:   Count occurrences of each skill
        4. $sort:    Order by count descending
        5. $limit:   Return top N skills

        $unwind is the key operator here:
        Before: {file_id: "001", extracted_skills: ["python", "docker", "aws"]}
        After:  {file_id: "001", skill: "python"}
                {file_id: "001", skill: "docker"}
                {file_id: "001", skill: "aws"}

        Each array element becomes a separate document that $group can count.

        Args:
            db:    Async MongoDB database
            limit: Number of top skills to return

        Returns:
            List of {skill, count, percentage} dicts for bar chart
        """
        total_resumes = await db["resumes"].count_documents({})

        pipeline = [
            # Stage 1: Keep only the skills array field
            # Reduces document size for faster pipeline processing
            {"$project": {"extracted_skills": 1, "_id": 0}},

            # Stage 2: Deconstruct skills array
            # preserveNullAndEmptyArrays: false = skip docs with empty arrays
            {"$unwind": {
                "path": "$extracted_skills",
                "preserveNullAndEmptyArrays": False
            }},

            # Stage 3: Group by skill name, count occurrences
            # _id: "$extracted_skills" = group key is the skill name
            {"$group": {
                "_id":   "$extracted_skills",  # Group key = skill name
                "count": {"$sum": 1}           # Count docs in each group
            }},

            # Stage 4: Sort by count descending
            {"$sort": {"count": -1}},

            # Stage 5: Take top N results
            {"$limit": limit},

            # Stage 6: Rename _id to skill for cleaner output
            {"$project": {
                "_id":  0,
                "skill": "$_id",
                "count": 1,
            }}
        ]

        results = await db["resumes"].aggregate(pipeline).to_list(length=limit)

        # Add percentage of resumes containing each skill
        for item in results:
            item["percentage"] = round(
                (item["count"] / total_resumes * 100) if total_resumes > 0 else 0,
                1
            )

        return results

    async def candidate_pipeline_summary(
        self,
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Any]:
        """
        Computes a summary of candidate counts at each pipeline stage.

        Pipeline stages (funnel):
        uploaded → nlp_processed → skills_extracted → scored → ranked

        Visualized as a funnel chart in the dashboard.
        Shows where candidates are dropping out of the pipeline —
        useful for identifying processing bottlenecks.

        Args:
            db: Async MongoDB database

        Returns:
            Dict with counts per stage for funnel chart
        """
        pipeline = [
            # Group all documents and count by status
            {"$group": {
                "_id":   "$status",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]

        results = await db["resumes"].aggregate(pipeline).to_list(length=20)

        # Convert to dict for easy lookup
        status_counts = {r["_id"]: r["count"] for r in results}

        # Scoring count comes from scores collection
        scored_count = await db["scores"].count_documents({})

        # Ranked count comes from rankings collection (unique file_ids)
        # Using distinct() to count unique candidates in rankings
        ranked_ids = await db["rankings"].distinct("ranked_candidates.file_id")
        ranked_count = len(ranked_ids)

        total = sum(status_counts.values())

        return {
            "funnel_stages": [
                {
                    "stage":      "Uploaded",
                    "count":      status_counts.get("uploaded", 0),
                    "percentage": self._pct(status_counts.get("uploaded", 0), total)
                },
                {
                    "stage":      "NLP Processed",
                    "count":      status_counts.get("nlp_processed", 0),
                    "percentage": self._pct(status_counts.get("nlp_processed", 0), total)
                },
                {
                    "stage":      "Skills Extracted",
                    "count":      status_counts.get("skills_extracted", 0),
                    "percentage": self._pct(status_counts.get("skills_extracted", 0), total)
                },
                {
                    "stage":      "Scored",
                    "count":      scored_count,
                    "percentage": self._pct(scored_count, total)
                },
                {
                    "stage":      "Ranked",
                    "count":      ranked_count,
                    "percentage": self._pct(ranked_count, total)
                },
            ],
            "total_resumes": total,
        }

    async def skill_gap_heatmap(
        self,
        db: AsyncIOMotorDatabase,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Computes skill gap data for all candidates scored against a job.

        For each required skill, calculates:
        - How many candidates have it (coverage)
        - How many are missing it (gap)
        - Coverage percentage

        Visualized as a horizontal bar chart in the dashboard.
        Recruiters see: "45% of candidates have Kubernetes — it's a gap."

        Aggregation:
        1. $match: scores for this job
        2. $project: extract skill_gap.matched and skill_gap.missing
        3. $unwind: one doc per missing skill
        4. $group: count how many candidates are missing each skill

        Args:
            db:     Async MongoDB database
            job_id: Job to compute gap analysis for

        Returns:
            Dict with skill gap data for heatmap visualization
        """

        # Pipeline for matched skills coverage
        matched_pipeline = [
            {"$match": {"job_id": job_id}},
            {"$project": {"matched": "$skill_gap.matched", "_id": 0}},
            {"$unwind": {"path": "$matched", "preserveNullAndEmptyArrays": False}},
            {"$group": {
                "_id":             "$matched",
                "candidate_count": {"$sum": 1}
            }},
            {"$sort": {"candidate_count": -1}},
        ]

        # Pipeline for missing skills (gaps)
        missing_pipeline = [
            {"$match": {"job_id": job_id}},
            {"$project": {"missing": "$skill_gap.missing", "_id": 0}},
            {"$unwind": {"path": "$missing", "preserveNullAndEmptyArrays": False}},
            {"$group": {
                "_id":             "$missing",
                "candidate_count": {"$sum": 1}
            }},
            {"$sort": {"candidate_count": -1}},
        ]

        import asyncio
        matched_results, missing_results = await asyncio.gather(
            db["scores"].aggregate(matched_pipeline).to_list(length=50),
            db["scores"].aggregate(missing_pipeline).to_list(length=50),
        )

        total_candidates = await db["scores"].count_documents({"job_id": job_id})

        # Build coverage data
        coverage = []
        for item in matched_results:
            coverage.append({
                "skill":      item["_id"],
                "have_count": item["candidate_count"],
                "coverage_pct": self._pct(item["candidate_count"], total_candidates),
            })

        # Build gap data
        gaps = []
        for item in missing_results:
            gaps.append({
                "skill":      item["_id"],
                "miss_count": item["candidate_count"],
                "gap_pct":    self._pct(item["candidate_count"], total_candidates),
            })

        return {
            "job_id":           job_id,
            "total_candidates": total_candidates,
            "skill_coverage":   coverage[:15],
            "skill_gaps":       gaps[:15],
        }

    async def hiring_funnel_metrics(
        self,
        db: AsyncIOMotorDatabase,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Computes hiring funnel conversion metrics.

        Funnel:
        Total Scored → Passed Threshold → Priority Tier → (Hired — external)

        Conversion rates between stages help recruiters identify
        whether the job description is too strict (low conversion)
        or too lenient (too many passing threshold).

        Args:
            db:     Async MongoDB database
            job_id: Specific job (None = all jobs)

        Returns:
            Dict with funnel metrics and conversion rates
        """
        match_filter = {"job_id": job_id} if job_id else {}

        pipeline = [
            {"$match": match_filter},
            {"$group": {
                "_id": None,
                "total_scored": {"$sum": 1},
                "passed_threshold": {
                    "$sum": {"$cond": ["$passes_threshold", 1, 0]}
                },
                "priority_tier": {
                    "$sum": {
                        "$cond": [{"$gte": ["$final_score", 0.80]}, 1, 0]
                    }
                },
                "avg_score": {"$avg": "$final_score"},
            }},
        ]

        results = await db["scores"].aggregate(pipeline).to_list(length=1)

        if not results:
            return {"total_scored": 0, "conversion_rates": {}}

        data = results[0]
        total    = data.get("total_scored", 0)
        passed   = data.get("passed_threshold", 0)
        priority = data.get("priority_tier", 0)

        return {
            "job_id":        job_id or "all",
            "total_scored":  total,
            "passed_threshold": passed,
            "priority_tier": priority,
            "avg_score":     round(data.get("avg_score", 0) * 100, 1),
            "conversion_rates": {
                "scored_to_threshold": self._pct(passed, total),
                "threshold_to_priority": self._pct(priority, passed),
            }
        }

    def _pct(self, numerator: int, denominator: int) -> float:
        """Safe percentage computation that handles zero denominator."""
        if denominator == 0:
            return 0.0
        return round(numerator / denominator * 100, 1)


# Single shared instance
aggregation_service = AggregationService()
