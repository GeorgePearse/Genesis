# PostgreSQL Integration for Genesis

This document describes the PostgreSQL integration for the Genesis Evolution Platform.

> **Note**: Genesis is based on [Shinka AI](https://github.com/shinkadotai/shinka) and maintains the original `genesis` package structure internally.

## Overview

Genesis now supports PostgreSQL as a database backend in addition to SQLite. The PostgreSQL support is implemented through Docker Compose and includes:

1. Database adapter layer (`genesis/database/adapter.py`)
2. PostgreSQL initialization script (`docker/init-db.sql`)
3. Docker Compose configuration with PostgreSQL service
4. Environment variable configuration for database selection

## Current Status

**⚠️ Work in Progress**: This is an initial implementation that provides the foundation for PostgreSQL support. The core Genesis codebase (`genesis/database/dbase.py`) still uses SQLite directly for most operations.

### What Works

✅ PostgreSQL container setup with Docker Compose
✅ Database initialization with proper schema
✅ Database adapter abstraction layer
✅ Environment-based database selection
✅ Connection management for both SQLite and PostgreSQL

### What Needs Work

🔧 Full integration of adapter layer throughout `dbase.py` (2000+ lines)
🔧 Migration of SQLite-specific SQL to database-agnostic queries
🔧 Handling of JSON columns (TEXT in SQLite vs JSONB in PostgreSQL)
🔧 Testing with actual evolution workloads
🔧 Data migration tools from SQLite to PostgreSQL

## Architecture

### Database Adapter Layer

The `DatabaseAdapter` abstract class in `genesis/database/adapter.py` provides a unified interface:

```python
from genesis.database.adapter import get_database_adapter

# Automatically selects adapter based on environment
adapter = get_database_adapter(
    database_url=os.getenv("DATABASE_URL"),  # PostgreSQL
    db_path="/path/to/sqlite.db"             # SQLite fallback
)

# Unified API regardless of backend
adapter.connect()
adapter.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
rows = adapter.fetchall()
```

### Schema Design

PostgreSQL schema uses JSONB for structured data:

| SQLite Type | PostgreSQL Type | Notes |
|-------------|----------------|-------|
| TEXT (JSON) | JSONB | Better performance, indexing support |
| REAL | DOUBLE PRECISION | Standard float type |
| INTEGER | INTEGER / SERIAL | AUTO INCREMENT → SERIAL |
| BOOLEAN | BOOLEAN | Native boolean support |

## Usage

### Docker Compose Setup

The PostgreSQL database is automatically configured in `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: genesis
      POSTGRES_USER: genesis
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
```

### Environment Variables

Configure the database backend via `.env`:

```bash
# PostgreSQL (recommended for Docker)
DATABASE_URL=postgresql://genesis:password@postgres:5432/genesis

# SQLite (default fallback)
# DATABASE_URL is not set, uses GENESIS_DATA_DIR/evolution.db
```

### Running with PostgreSQL

```bash
# Copy and configure environment
cp .env.example .env
# Set DATABASE_URL in .env

# Start services
docker-compose up -d

# Check PostgreSQL logs
docker-compose logs -f postgres

# Access PostgreSQL directly
docker-compose exec postgres psql -U genesis -d genesis
```

## Database Schema

The PostgreSQL schema is defined in `docker/init-db.sql`:

### Tables

**programs**: Core table storing evolved programs
- Primary key: `id` (TEXT)
- JSON fields: `archive_inspiration_ids`, `top_k_inspiration_ids`, `public_metrics`, `private_metrics`, `metadata`, `embedding`, `migration_history`
- Indices on: `generation`, `timestamp`, `complexity`, `parent_id`, `island_idx`, `correct`, `combined_score`

**archive**: Elite program storage
- Foreign key to `programs(id)`

**metadata_store**: Evolution state tracking
- Key-value store for runtime metadata

## Migration Path

### Phase 1: Foundation (Current)
- ✅ Database adapter layer
- ✅ PostgreSQL container setup
- ✅ Schema initialization
- ✅ Environment configuration

### Phase 2: Integration (Future)
- 🔧 Refactor `ProgramDatabase` to use adapter
- 🔧 Replace direct `sqlite3` calls with adapter methods
- 🔧 Handle SQL dialect differences (? vs %s placeholders)
- 🔧 Update query builders for PostgreSQL compatibility

### Phase 3: Optimization (Future)
- 🔧 JSONB indexing for performance
- 🔧 Connection pooling
- 🔧 Query optimization
- 🔧 Batch operations

### Phase 4: Production (Future)
- 🔧 Data migration tools (SQLite → PostgreSQL)
- 🔧 Backup and restore procedures
- 🔧 Monitoring and observability
- 🔧 High availability setup

## Contributing

To contribute to PostgreSQL integration:

1. **Understand the adapter pattern**: Review `genesis/database/adapter.py`
2. **Identify SQLite-specific code**: Look for `sqlite3` imports and SQL dialect specifics in `dbase.py`
3. **Refactor incrementally**: Replace direct database calls with adapter methods
4. **Test thoroughly**: Ensure both SQLite and PostgreSQL work
5. **Document changes**: Update this guide with your improvements

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `genesis/database/adapter.py` | Database abstraction layer | ✅ Complete |
| `genesis/database/dbase.py` | Core database operations | 🔧 Needs refactoring |
| `docker/init-db.sql` | PostgreSQL schema | ✅ Complete |
| `docker-compose.yml` | Container orchestration | ✅ Complete |
| `pyproject.toml` | Dependencies (psycopg) | ✅ Complete |

## Troubleshooting

### Connection Errors

```
psycopg.OperationalError: could not connect to server
```

**Solution**: Ensure PostgreSQL is healthy:
```bash
docker-compose ps
docker-compose logs postgres
```

### Schema Errors

```
relation "programs" does not exist
```

**Solution**: Check if init script ran:
```bash
docker-compose exec postgres psql -U genesis -d genesis -c "\dt"
```

If tables are missing, recreate the database:
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Restart with fresh database
```

### Adapter Not Used

```
# Current behavior: Still using SQLite
```

**Expected**: The current implementation doesn't fully integrate the adapter. This is intentional - the adapter layer is a foundation for future work. To use PostgreSQL today, you would need to refactor `dbase.py` to use the adapter methods.

## Performance Considerations

### PostgreSQL Advantages
- Better concurrency (no locking issues)
- JSONB performance and indexing
- Full ACID compliance
- Better suited for multi-user scenarios
- Advanced query optimization

### SQLite Advantages
- Zero configuration
- Single-file portability
- Lower resource usage
- Perfect for single-user local development
- Simpler deployment

## Future Enhancements

1. **Connection Pooling**: Use psycopg's pool for better performance
2. **Async Support**: Leverage psycopg3's async capabilities
3. **Migrations**: Add Alembic for schema versioning
4. **Replication**: PostgreSQL streaming replication for HA
5. **Monitoring**: Integration with pgAdmin or similar tools
6. **Partitioning**: Table partitioning for large datasets
7. **Full-text Search**: Leverage PostgreSQL's FTS for code search

## References

- [psycopg3 Documentation](https://www.psycopg.org/psycopg3/)
- [PostgreSQL JSON Types](https://www.postgresql.org/docs/current/datatype-json.html)
- [Docker PostgreSQL](https://hub.docker.com/_/postgres)
- [SQLite vs PostgreSQL](https://www.sqlite.org/whentouse.html)

## Questions?

For issues or questions about PostgreSQL integration:
1. Check [DOCKER.md](DOCKER.md) for Docker setup
2. Review this document for PostgreSQL-specific information
3. Open an issue on GitHub with details about your problem
