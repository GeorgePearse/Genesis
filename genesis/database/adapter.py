"""
Database adapter layer for SQLite and PostgreSQL support.

This module provides a unified interface for database operations that works
with both SQLite and PostgreSQL backends.
"""

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""

    @abstractmethod
    def connect(self) -> Any:
        """Establish database connection."""
        pass

    @abstractmethod
    def cursor(self) -> Any:
        """Get a database cursor."""
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query with optional parameters."""
        pass

    @abstractmethod
    def executemany(self, query: str, params_list: List[Tuple]) -> Any:
        """Execute a query with multiple parameter sets."""
        pass

    @abstractmethod
    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row as a dictionary."""
        pass

    @abstractmethod
    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def get_placeholder(self) -> str:
        """Get the parameter placeholder character (? or %s)."""
        pass

    @abstractmethod
    def adapt_create_table_sql(self, sql: str) -> str:
        """Adapt CREATE TABLE SQL for the specific database."""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter."""

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self.conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

    def connect(self) -> sqlite3.Connection:
        """Establish SQLite connection."""
        if self.conn:
            return self.conn

        if self.db_path:
            db_file = Path(self.db_path).resolve()
            if self.read_only:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"file:{db_file}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
            else:
                # Robustness check for unclean shutdown with WAL
                db_wal_file = Path(f"{db_file}-wal")
                db_shm_file = Path(f"{db_file}-shm")
                if (
                    db_file.exists()
                    and db_file.stat().st_size == 0
                    and (db_wal_file.exists() or db_shm_file.exists())
                ):
                    logger.warning(
                        f"Database file {db_file} is empty but WAL/SHM files exist. "
                        "Removing WAL/SHM files to attempt recovery."
                    )
                    if db_wal_file.exists():
                        db_wal_file.unlink()
                    if db_shm_file.exists():
                        db_shm_file.unlink()
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(db_file), timeout=30.0)
        else:
            self.conn = sqlite3.connect(":memory:")

        self.conn.row_factory = sqlite3.Row
        return self.conn

    def cursor(self) -> sqlite3.Cursor:
        """Get a SQLite cursor."""
        if not self.conn:
            self.connect()
        if not self._cursor:
            self._cursor = self.conn.cursor()
        return self._cursor

    def execute(self, query: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """Execute a query with optional parameters."""
        cursor = self.cursor()
        if params:
            return cursor.execute(query, params)
        return cursor.execute(query)

    def executemany(self, query: str, params_list: List[Tuple]) -> sqlite3.Cursor:
        """Execute a query with multiple parameter sets."""
        cursor = self.cursor()
        return cursor.executemany(query, params_list)

    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row as a dictionary."""
        row = self.cursor().fetchone()
        return dict(row) if row else None

    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        rows = self.cursor().fetchall()
        return [dict(row) for row in rows]

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()

    def close(self) -> None:
        """Close the database connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_placeholder(self) -> str:
        """Get the parameter placeholder character."""
        return "?"

    def adapt_create_table_sql(self, sql: str) -> str:
        """SQLite doesn't need adaptation."""
        return sql


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn: Optional[Any] = None
        self._cursor: Optional[Any] = None

        # Import psycopg3 only when needed
        try:
            import psycopg
            from psycopg.rows import dict_row
            self.psycopg = psycopg
            self.dict_row = dict_row
        except ImportError:
            raise ImportError(
                "psycopg is required for PostgreSQL support. "
                "Install it with: pip install 'psycopg[binary]'"
            )

    def connect(self) -> Any:
        """Establish PostgreSQL connection."""
        if self.conn:
            return self.conn

        self.conn = self.psycopg.connect(
            self.database_url,
            row_factory=self.dict_row
        )
        return self.conn

    def cursor(self) -> Any:
        """Get a PostgreSQL cursor."""
        if not self.conn:
            self.connect()
        if not self._cursor:
            self._cursor = self.conn.cursor()
        return self._cursor

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query with optional parameters."""
        cursor = self.cursor()
        # Convert ? placeholders to %s for PostgreSQL
        query = query.replace("?", "%s")
        if params:
            return cursor.execute(query, params)
        return cursor.execute(query)

    def executemany(self, query: str, params_list: List[Tuple]) -> Any:
        """Execute a query with multiple parameter sets."""
        cursor = self.cursor()
        # Convert ? placeholders to %s for PostgreSQL
        query = query.replace("?", "%s")
        return cursor.executemany(query, params_list)

    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row as a dictionary."""
        row = self.cursor().fetchone()
        return dict(row) if row else None

    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        rows = self.cursor().fetchall()
        return [dict(row) for row in rows]

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()

    def close(self) -> None:
        """Close the database connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_placeholder(self) -> str:
        """Get the parameter placeholder character."""
        return "%s"

    def adapt_create_table_sql(self, sql: str) -> str:
        """Adapt SQLite SQL for PostgreSQL."""
        # Replace SQLite-specific syntax with PostgreSQL equivalents
        sql = sql.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
        sql = sql.replace("AUTOINCREMENT", "")
        sql = sql.replace("REAL", "DOUBLE PRECISION")
        sql = sql.replace("BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE")
        sql = sql.replace("BOOLEAN DEFAULT 1", "BOOLEAN DEFAULT TRUE")

        # Handle PRAGMA statements (not needed in PostgreSQL)
        if sql.strip().startswith("PRAGMA"):
            return ""

        return sql


def get_database_adapter(
    db_path: Optional[str] = None,
    database_url: Optional[str] = None,
    read_only: bool = False
) -> DatabaseAdapter:
    """
    Factory function to get the appropriate database adapter.

    Args:
        db_path: Path to SQLite database file (for SQLite)
        database_url: PostgreSQL connection URL (for PostgreSQL)
        read_only: Whether to open in read-only mode (SQLite only)

    Returns:
        DatabaseAdapter: The appropriate database adapter instance

    Raises:
        ValueError: If neither db_path nor database_url is provided, or both are provided
    """
    # Check environment variable for DATABASE_URL
    if not database_url:
        database_url = os.environ.get("DATABASE_URL")

    # Determine which adapter to use
    if database_url:
        if database_url.startswith(("postgres://", "postgresql://")):
            logger.info(f"Using PostgreSQL adapter")
            return PostgreSQLAdapter(database_url)
        else:
            raise ValueError(f"Unsupported database URL scheme: {database_url}")
    elif db_path:
        logger.info(f"Using SQLite adapter with db_path: {db_path}")
        return SQLiteAdapter(db_path, read_only)
    else:
        # Default to in-memory SQLite
        logger.info("Using in-memory SQLite adapter")
        return SQLiteAdapter(None, read_only)
