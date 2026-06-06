

"""
Master skill taxonomy for the AI Career Intelligence Platform.

Structure:
    SKILL_TAXONOMY: Dict[category_name, List[canonical_skill_names]]
    SKILL_ALIASES:  Dict[alias/abbreviation, canonical_name]
    CATEGORY_WEIGHTS: Dict[category_name, float] — importance multiplier

Design principles:
    1. Canonical names use the most common industry spelling
    2. All canonical names are lowercase for case-insensitive matching
    3. Aliases cover abbreviations, typos, and alternate spellings
    4. Categories reflect how recruiters think about skill domains
"""

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: MASTER SKILL TAXONOMY
# ─────────────────────────────────────────────────────────────────────
# Each key is a category name used for grouping and UI display.
# Each value is a list of canonical skill strings in lowercase.
# Multi-word skills ("machine learning") are intentionally included
# because the extractor checks bigrams and trigrams, not just tokens.

SKILL_TAXONOMY = {

    # ── Programming Languages ──────────────────────────────────────
    # Core programming languages — highest signal for technical roles
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c", "c++", "c#",
        "go", "golang", "rust", "ruby", "php", "swift", "kotlin",
        "scala", "r", "matlab", "perl", "bash", "shell", "powershell",
        "objective-c", "dart", "lua", "haskell", "erlang", "elixir",
        "clojure", "f#", "groovy", "assembly", "cobol", "fortran",
        "vba", "sql", "plsql", "tsql",
    ],

    # ── Web & API Frameworks ───────────────────────────────────────
    # Frameworks are strong differentiators for backend and frontend roles
    "frameworks_and_libraries": [
        # Python web
        "fastapi", "django", "flask", "tornado", "starlette", "aiohttp",
        # JavaScript/TypeScript web
        "react", "react.js", "angular", "vue", "vue.js", "next.js",
        "nuxt.js", "express", "express.js", "node.js", "nestjs",
        "svelte", "ember.js", "backbone.js",
        # Java/JVM
        "spring", "spring boot", "spring mvc", "hibernate", "struts",
        "micronaut", "quarkus",
        # Ruby
        "rails", "ruby on rails", "sinatra",
        # PHP
        "laravel", "symfony", "wordpress",
        # Mobile
        "react native", "flutter", "xamarin",
        # Testing
        "pytest", "unittest", "jest", "mocha", "cypress", "selenium",
        "playwright", "junit", "testng",
        # Utilities
        "celery", "pydantic", "sqlalchemy", "alembic",
        "graphql", "rest api", "restful api", "soap",
    ],

    # ── Data Science & Machine Learning ───────────────────────────
    # Fastest-growing category — high salary premium
    "data_science_and_ml": [
        # Core ML
        "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "transfer learning",
        "supervised learning", "unsupervised learning",
        # Frameworks
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "transformers",
        "opencv", "spacy", "nltk", "gensim",
        # Data tools
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "jupyter", "jupyter notebook", "conda", "anaconda",
        # MLOps
        "mlflow", "kubeflow", "airflow", "apache airflow", "prefect",
        "dvc", "wandb", "weights and biases",
        # Concepts
        "feature engineering", "model deployment", "a/b testing",
        "statistical analysis", "data analysis", "data science",
        "data mining", "data visualization", "big data",
        "time series", "regression", "classification", "clustering",
        "dimensionality reduction", "pca",
    ],

    # ── Databases & Storage ─────────────────────────────────────────
    # Every backend role requires database skills
    "databases": [
        # Relational
        "postgresql", "mysql", "sqlite", "oracle", "sql server",
        "mariadb", "db2",
        # Document
        "mongodb", "couchdb", "firestore", "dynamodb",
        # Key-value / Cache
        "redis", "memcached", "etcd",
        # Column-family
        "cassandra", "hbase", "bigtable",
        # Search
        "elasticsearch", "opensearch", "solr",
        # Graph
        "neo4j", "janusgraph", "dgraph",
        # Time-series
        "influxdb", "timescaledb", "prometheus",
        # Data warehouse
        "snowflake", "bigquery", "redshift", "databricks",
        "apache spark", "spark", "hadoop", "hive", "presto", "dbt",
        # Concepts
        "database design", "data modeling", "orm", "query optimization",
        "indexing", "replication", "sharding", "acid",
    ],

    # ── Cloud & DevOps ──────────────────────────────────────────────
    # Infrastructure skills — increasingly mandatory for senior roles
    "cloud_and_devops": [
        # Cloud platforms
        "aws", "amazon web services", "gcp", "google cloud",
        "azure", "microsoft azure", "alibaba cloud", "oracle cloud",
        # AWS services
        "ec2", "s3", "lambda", "rds", "ecs", "eks", "cloudformation",
        "api gateway", "sqs", "sns", "cloudwatch", "iam", "vpc",
        # GCP services
        "gke", "cloud run", "cloud functions", "pub/sub", "dataflow",
        # Azure services
        "azure devops", "azure functions", "aks",
        # Containers & Orchestration
        "docker", "kubernetes", "k8s", "helm", "docker compose",
        "podman", "openshift",
        # CI/CD
        "jenkins", "github actions", "gitlab ci", "circleci",
        "travis ci", "argocd", "spinnaker",
        # Infrastructure as Code
        "terraform", "ansible", "puppet", "chef", "cloudformation",
        "pulumi",
        # Monitoring & Observability
        "grafana", "kibana", "datadog", "new relic", "splunk",
        "prometheus", "jaeger", "opentelemetry",
        # Networking
        "nginx", "apache", "load balancing", "cdn", "dns",
        "microservices", "service mesh", "istio", "consul",
        "message queue", "kafka", "rabbitmq", "zeromq",
    ],

    # ── Software Engineering Practices ─────────────────────────────
    # Process and methodology skills — show seniority and teamwork
    "engineering_practices": [
        # Development practices
        "agile", "scrum", "kanban", "tdd", "test driven development",
        "bdd", "behavior driven development", "pair programming",
        "code review", "technical leadership",
        # Architecture
        "system design", "software architecture", "microservices",
        "event driven architecture", "domain driven design", "ddd",
        "rest", "api design", "clean architecture", "solid",
        "design patterns",
        # Version control
        "git", "github", "gitlab", "bitbucket", "svn",
        # Security
        "cybersecurity", "oauth", "jwt", "ssl", "tls", "encryption",
        "penetration testing", "sast", "owasp",
        # Other
        "linux", "unix", "windows server",
        "project management", "technical writing", "documentation",
        "open source",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: SKILL ALIASES
# ─────────────────────────────────────────────────────────────────────
# Maps non-canonical forms → canonical taxonomy entries.
#
# Coverage:
#   - Common abbreviations:    "ml" → "machine learning"
#   - Alternate capitalizations: "javascript" matches "JavaScript" via lower()
#   - Version suffixes:        "python3" → "python"
#   - Vendor-specific names:   "postgres" → "postgresql"
#   - Common typos:            "kubernets" → "kubernetes"
#   - Short forms:             "k8s" → "kubernetes"
#   - Compound variations:     "node" → "node.js"

SKILL_ALIASES = {
    # Programming language aliases
    "py":           "python",
    "python3":      "python",
    "python2":      "python",
    "js":           "javascript",
    "es6":          "javascript",
    "es2015":       "javascript",
    "ts":           "typescript",
    "node":         "node.js",
    "nodejs":       "node.js",
    "golang":       "go",
    "cplusplus":    "c++",
    "csharp":       "c#",
    "dotnet":       "c#",
    ".net":         "c#",
    "rb":           "ruby",

    # Framework aliases
    "reactjs":      "react",
    "react js":     "react",
    "vuejs":        "vue",
    "angularjs":    "angular",
    "nextjs":       "next.js",
    "expressjs":    "express",
    "springboot":   "spring boot",
    "rails":        "ruby on rails",
    "ror":          "ruby on rails",
    "sklearn":      "scikit-learn",

    # Database aliases
    "postgres":     "postgresql",
    "postgre":      "postgresql",
    "postgresSQL":  "postgresql",
    "mongo":        "mongodb",
    "mssql":        "sql server",
    "ms sql":       "sql server",
    "elastic":      "elasticsearch",
    "es":           "elasticsearch",
    "dynamo":       "dynamodb",

    # Cloud aliases
    "amazon aws":   "aws",
    "gcloud":       "gcp",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "k8s":          "kubernetes",
    "kube":         "kubernetes",
    "kubernets":    "kubernetes",  # common typo
    "kuberntes":    "kubernetes",  # common typo

    # ML aliases
    "ml":           "machine learning",
    "dl":           "deep learning",
    "nlp":          "natural language processing",
    "cv":           "computer vision",
    "ai":           "machine learning",
    "tf":           "tensorflow",
    "pytorch":      "pytorch",
    "hf":           "hugging face",
    "xgb":          "xgboost",
    "lgbm":         "lightgbm",

    # DevOps aliases
    "gh actions":   "github actions",
    "gha":          "github actions",
    "gitlab-ci":    "gitlab ci",
    "iac":          "terraform",
    "infra as code": "terraform",
    "ci/cd":        "jenkins",  # map generic term to category concept
    "cicd":         "jenkins",

    # General
    "oop":          "design patterns",
    "object oriented": "design patterns",
    "rest":         "rest api",
    "restful":      "restful api",
    "sql":          "postgresql",  # generic SQL maps to most common RDBMS
    "nosql":        "mongodb",
    "vcs":          "git",
}


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: CATEGORY WEIGHTS
# ─────────────────────────────────────────────────────────────────────
# Multipliers applied to skill scores during ATS scoring.
# Based on typical importance weighting in technical job descriptions.
# These can be overridden per job posting in Module 5.

CATEGORY_WEIGHTS = {
    "programming_languages":   1.5,   # Highest: core requirement for any tech role
    "frameworks_and_libraries": 1.4,  # Very high: specific stack knowledge
    "data_science_and_ml":     1.3,   # High: specialized, high-value skills
    "databases":               1.2,   # Medium-high: universal backend requirement
    "cloud_and_devops":        1.2,   # Medium-high: infrastructure increasingly required
    "engineering_practices":   1.0,   # Baseline: methodology and soft skills
}


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: DERIVED DATA STRUCTURES (computed at import time)
# ─────────────────────────────────────────────────────────────────────

# Flat set of ALL canonical skills across all categories
# Used for O(1) membership testing during extraction
ALL_SKILLS: set = set()
for category, skills in SKILL_TAXONOMY.items():
    ALL_SKILLS.update(skills)

# Reverse lookup: canonical_skill → category_name
# Example: "python" → "programming_languages"
SKILL_TO_CATEGORY: dict = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill] = category

# Set of all multi-word skills (length > 1 when split)
# Used to avoid false bigram matches against single-word skill names
MULTI_WORD_SKILLS: set = {
    skill for skill in ALL_SKILLS
    if len(skill.split()) > 1
}

# Set of all single-word skills
SINGLE_WORD_SKILLS: set = ALL_SKILLS - MULTI_WORD_SKILLS

# Complete lookup including aliases → canonical
# Merges ALL_SKILLS (self-mapping) with SKILL_ALIASES
COMPLETE_SKILL_LOOKUP: dict = {skill: skill for skill in ALL_SKILLS}
COMPLETE_SKILL_LOOKUP.update(SKILL_ALIASES)
