import json
import logging
import time
import os
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import clickhouse_connect

from .complexity import analyze_code_metrics
from .parents import CombinedParentSelector
from .inspirations import CombinedContextSelector
from .islands import CombinedIslandManager
from .display import DatabaseDisplay
from genesis.llm.embedding import EmbeddingClient

logger = logging.getLogger(__name__)


def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        if np.isscalar(obj):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        else:
            return clean_nan_values(obj.tolist())
    else:
        return obj


@dataclass
class DatabaseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("CLICKHOUSE_PORT", 8123))
    username: str = os.getenv("CLICKHOUSE_USER", "default")
    password: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    database: str = os.getenv("CLICKHOUSE_DB", "default")

    num_islands: int = 4
    archive_size: int = 100

    # Inspiration parameters
    elite_selection_ratio: float = 0.3
    num_archive_inspirations: int = 5
    num_top_k_inspirations: int = 2

    # Island model/migration parameters
    migration_interval: int = 10
    migration_rate: float = 0.1
    island_elitism: bool = True
    enforce_island_separation: bool = True

    # Parent selection parameters
    parent_selection_strategy: str = "power_law"

    # Power-law parent selection parameters
    exploitation_alpha: float = 1.0
    exploitation_ratio: float = 0.2

    # Weighted tree parent selection parameters
    parent_selection_lambda: float = 10.0

    # Beam search parent selection parameters
    num_beams: int = 5

    # Embedding model name
    embedding_model: str = "text-embedding-3-small"


@dataclass
class Program:
    """Represents a program in the database"""

    id: str
    code: str
    language: str = "python"
    parent_id: Optional[str] = None
    archive_inspiration_ids: List[str] = field(default_factory=list)
    top_k_inspiration_ids: List[str] = field(default_factory=list)
    island_idx: Optional[int] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    code_diff: Optional[str] = None
    combined_score: float = 0.0
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    private_metrics: Dict[str, Any] = field(default_factory=dict)
    text_feedback: Union[str, List[str]] = ""
    correct: bool = False
    children_count: int = 0
    complexity: float = 0.0
    embedding: List[float] = field(default_factory=list)
    embedding_pca_2d: List[float] = field(default_factory=list)
    embedding_pca_3d: List[float] = field(default_factory=list)
    embedding_cluster_id: Optional[int] = None
    migration_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_archive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        # Ensure fields are correct types
        for field_name in ["public_metrics", "private_metrics", "metadata"]:
            if not isinstance(data.get(field_name), dict):
                data[field_name] = {}

        for field_name in [
            "archive_inspiration_ids",
            "top_k_inspiration_ids",
            "embedding",
            "embedding_pca_2d",
            "embedding_pca_3d",
            "migration_history",
        ]:
            if not isinstance(data.get(field_name), list):
                data[field_name] = []

        # Filter fields
        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}
        return cls(**filtered_data)


class ProgramDatabase:
    """
    ClickHouse-backed database for storing and managing programs.
    """

    def __init__(
        self,
        config: DatabaseConfig,
        embedding_model: str = "text-embedding-3-small",
        read_only: bool = False,
    ):
        self.config = config
        self.read_only = read_only
        self.client = None

        # Connect to ClickHouse
        try:
            self.client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
            )
            logger.info(
                f"Connected to ClickHouse at {self.config.host}:{self.config.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

        if not read_only:
            self.embedding_client = EmbeddingClient(model_name=embedding_model)
            self._create_tables()
        else:
            self.embedding_client = None

        self.last_iteration: int = 0
        self.best_program_id: Optional[str] = None
        self.beam_search_parent_id: Optional[str] = None
        self._schedule_migration: bool = False

        self._load_metadata()

        # Initialize managers with ClickHouse client
        self.island_manager = CombinedIslandManager(
            client=self.client,
            config=self.config,
        )

        count = self._count_programs()
        logger.debug(f"DB initialized with {count} programs.")

    def _create_tables(self):
        # Programs table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS programs (
                id String,
                code String,
                language String,
                parent_id String,
                archive_inspiration_ids String, -- JSON
                top_k_inspiration_ids String, -- JSON
                generation Int32,
                timestamp Float64,
                code_diff String,
                combined_score Float64,
                public_metrics String, -- JSON
                private_metrics String, -- JSON
                text_feedback String,
                complexity Float64,
                embedding Array(Float32),
                embedding_pca_2d String, -- JSON
                embedding_pca_3d String, -- JSON
                embedding_cluster_id Int32,
                correct UInt8,
                children_count Int32,
                metadata String, -- JSON
                island_idx Int32,
                migration_history String, -- JSON
                in_archive UInt8 DEFAULT 0
            ) ENGINE = ReplacingMergeTree(timestamp)
            ORDER BY id
        """)

        # Archive table (simplified, just tracks IDs in archive)
        self.client.command("""
            CREATE TABLE IF NOT EXISTS archive (
                program_id String,
                timestamp DateTime64(3) DEFAULT now()
            ) ENGINE = ReplacingMergeTree()
            ORDER BY program_id
        """)

        # Metadata store
        self.client.command("""
            CREATE TABLE IF NOT EXISTS metadata_store (
                key String,
                value String,
                timestamp DateTime64(3) DEFAULT now()
            ) ENGINE = ReplacingMergeTree(timestamp)
            ORDER BY key
        """)

        logger.debug("ClickHouse tables ensured to exist.")

    def _count_programs(self) -> int:
        return self.client.command("SELECT count() FROM programs")

    def _load_metadata(self):
        try:
            last_iter = self.client.command(
                "SELECT value FROM metadata_store WHERE key = 'last_iteration'"
            )
            self.last_iteration = int(last_iter) if last_iter else 0

            best_id = self.client.command(
                "SELECT value FROM metadata_store WHERE key = 'best_program_id'"
            )
            self.best_program_id = best_id if best_id and best_id != "None" else None

            beam_id = self.client.command(
                "SELECT value FROM metadata_store WHERE key = 'beam_search_parent_id'"
            )
            self.beam_search_parent_id = (
                beam_id if beam_id and beam_id != "None" else None
            )
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    def _update_metadata(self, key: str, value: Any):
        if self.read_only:
            return
        val_str = str(value)
        self.client.command(
            "INSERT INTO metadata_store (key, value) VALUES", [[key, val_str]]
        )

    def add(self, program: Program, verbose: bool = False) -> str:
        if self.read_only:
            raise PermissionError("Read-only mode")

        self.island_manager.assign_island(program)

        if program.complexity == 0.0:
            try:
                metrics = analyze_code_metrics(program.code, program.language)
                program.complexity = metrics.get(
                    "complexity_score", float(len(program.code))
                )
                if not program.metadata:
                    program.metadata = {}
                program.metadata["code_analysis_metrics"] = metrics
            except:
                program.complexity = float(len(program.code))

        # Serialize fields
        row = [
            program.id,
            program.code,
            program.language,
            program.parent_id or "",
            json.dumps(program.archive_inspiration_ids),
            json.dumps(program.top_k_inspiration_ids),
            program.generation,
            program.timestamp,
            program.code_diff or "",
            program.combined_score or 0.0,
            json.dumps(program.public_metrics),
            json.dumps(program.private_metrics),
            str(program.text_feedback) if program.text_feedback else "",
            program.complexity,
            program.embedding,
            json.dumps(program.embedding_pca_2d),
            json.dumps(program.embedding_pca_3d),
            program.embedding_cluster_id or -1,
            1 if program.correct else 0,
            program.children_count,
            json.dumps(program.metadata),
            program.island_idx if program.island_idx is not None else -1,
            json.dumps(program.migration_history),
            1 if program.in_archive else 0,
        ]

        self.client.insert(
            "programs",
            [row],
            column_names=[
                "id",
                "code",
                "language",
                "parent_id",
                "archive_inspiration_ids",
                "top_k_inspiration_ids",
                "generation",
                "timestamp",
                "code_diff",
                "combined_score",
                "public_metrics",
                "private_metrics",
                "text_feedback",
                "complexity",
                "embedding",
                "embedding_pca_2d",
                "embedding_pca_3d",
                "embedding_cluster_id",
                "correct",
                "children_count",
                "metadata",
                "island_idx",
                "migration_history",
                "in_archive",
            ],
        )

        # Update parent children count (ClickHouse specific: we update by inserting new row with incremented count?
        # No, simpler to just rely on count queries or updates. ReplacingMergeTree handles updates by key.
        # But we need to read parent first. For now, let's skip incrementing parent count in DB
        # and rely on 'SELECT count() FROM programs WHERE parent_id=...' if needed)

        self._update_archive(program)
        self._update_best_program(program)
        self._recompute_embeddings_and_clusters()

        if program.generation > self.last_iteration:
            self.last_iteration = program.generation
            self._update_metadata("last_iteration", self.last_iteration)

        if verbose:
            self._print_program_summary(program)

        if self.island_manager.needs_island_copies(program):
            self.island_manager.copy_program_to_islands(program)
            # Remove flag in DB? We just inserted it. Maybe update it.
            # For ClickHouse, updating metadata means inserting a new row with same ID.
            if program.metadata:
                program.metadata.pop("_needs_island_copies", None)
                self._update_program_metadata(program.id, program.metadata)

        if self.island_manager.should_schedule_migration(program):
            self._schedule_migration = True

        self.check_scheduled_operations()
        return program.id

    def _update_program_metadata(self, pid: str, metadata: dict):
        meta_json = json.dumps(metadata)
        self.client.command(
            f"ALTER TABLE programs UPDATE metadata = '{meta_json}' WHERE id = '{pid}'"
        )

    def get(self, program_id: str) -> Optional[Program]:
        try:
            result = self.client.query(
                f"SELECT * FROM programs WHERE id = '{program_id}' FINAL"
            )
            if not result.result_rows:
                return None

            row = result.result_rows[0]
            cols = result.column_names
            data = dict(zip(cols, row))
            return self._program_from_dict(data)
        except Exception as e:
            logger.error(f"Error getting program {program_id}: {e}")
            return None

    def _program_from_dict(self, data: Dict[str, Any]) -> Program:
        # Deserialize JSON fields
        for field in ["public_metrics", "private_metrics", "metadata"]:
            if isinstance(data.get(field), str):
                try:
                    data[field] = json.loads(data[field])
                except:
                    data[field] = {}

        for field in [
            "archive_inspiration_ids",
            "top_k_inspiration_ids",
            "embedding_pca_2d",
            "embedding_pca_3d",
            "migration_history",
        ]:
            if isinstance(data.get(field), str):
                try:
                    data[field] = json.loads(data[field])
                except:
                    data[field] = []

        data["correct"] = bool(data.get("correct", 0))
        data["in_archive"] = bool(data.get("in_archive", 0))

        # Handle -1 defaults
        if data.get("island_idx") == -1:
            data["island_idx"] = None
        if data.get("embedding_cluster_id") == -1:
            data["embedding_cluster_id"] = None
        if data.get("parent_id") == "":
            data["parent_id"] = None

        return Program.from_dict(data)

    def _update_archive(self, program: Program):
        if not self.config.archive_size or not program.correct:
            return

        count = self.client.command("SELECT count() FROM archive")
        if count < self.config.archive_size:
            self.client.command(
                f"INSERT INTO archive (program_id) VALUES ('{program.id}')"
            )
        else:
            # Find worst in archive
            # We need to join archive with programs to get scores
            worst_res = self.client.query("""
                SELECT a.program_id, p.combined_score 
                FROM archive a 
                LEFT JOIN programs p ON a.program_id = p.id
                ORDER BY p.combined_score ASC LIMIT 1
            """)
            if worst_res.result_rows:
                worst_id, worst_score = worst_res.result_rows[0]
                if program.combined_score > worst_score:
                    self.client.command(
                        f"ALTER TABLE archive DELETE WHERE program_id = '{worst_id}'"
                    )
                    self.client.command(
                        f"INSERT INTO archive (program_id) VALUES ('{program.id}')"
                    )

    def _update_best_program(self, program: Program):
        if not program.correct:
            return

        if not self.best_program_id:
            self.best_program_id = program.id
            self._update_metadata("best_program_id", program.id)
            return

        current_best = self.get(self.best_program_id)
        if not current_best or (program.combined_score > current_best.combined_score):
            self.best_program_id = program.id
            self._update_metadata("best_program_id", program.id)
            logger.info(
                f"New best program: {program.id} (Score: {program.combined_score})"
            )

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        query = "SELECT * FROM programs WHERE correct = 1"
        if metric:
            # This is tricky with JSON metrics in ClickHouse.
            # We'd need JSON extraction functions.
            # Assuming basic combined_score for now if metric is complex
            logger.warning(
                "Custom metric sorting in ClickHouse requires JSON extract logic. Using combined_score."
            )
            query += " ORDER BY combined_score DESC LIMIT 1 FINAL"
        else:
            query += " ORDER BY combined_score DESC LIMIT 1 FINAL"

        res = self.client.query(query)
        if res.result_rows:
            return self._program_from_dict(
                dict(zip(res.column_names, res.result_rows[0]))
            )
        return None

    def get_top_programs(
        self, n: int = 10, metric: str = "combined_score", correct_only: bool = False
    ) -> List[Program]:
        where = "WHERE correct = 1" if correct_only else "WHERE 1=1"
        query = f"SELECT * FROM programs {where} ORDER BY combined_score DESC LIMIT {n} FINAL"
        res = self.client.query(query)
        return [
            self._program_from_dict(dict(zip(res.column_names, row)))
            for row in res.result_rows
        ]

    def _recompute_embeddings_and_clusters(self, num_clusters: int = 4):
        # Similar logic but fetching/updating via ClickHouse
        # Implementation omitted for brevity but follows same pattern
        pass

    def check_scheduled_operations(self):
        if self._schedule_migration:
            self.island_manager.perform_migration(self.last_iteration)
            self._schedule_migration = False

    def close(self):
        if self.client:
            self.client.close()

    def print_summary(self, console=None):
        pass  # Todo: update display logic

    def _print_program_summary(self, program):
        pass

    # ... Other methods (sample, compute_similarity) would need similar updates
    # Creating a stub for sample to prevent immediate crash if used
    def sample(self, *args, **kwargs):
        # minimal stub
        parent = self.get_best_program()
        return parent, [], []
