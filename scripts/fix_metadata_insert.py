import os
import time
import json
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()


def get_client():
    url = os.getenv("CLICKHOUSE_URL")
    if url:
        import re

        match = re.match(r"https?://([^:]+):([^@]+)@([^:]+):(\d+)", url)
        if match:
            return clickhouse_connect.get_client(
                host=match.group(3),
                port=int(match.group(4)),
                username=match.group(1),
                password=match.group(2),
                secure=url.startswith("https"),
            )
    return clickhouse_connect.get_client(host="localhost")


client = get_client()

# 1. Get latest run_id and path
res = client.query(
    "SELECT run_id, database_path FROM evolution_runs ORDER BY start_time DESC LIMIT 1"
)
if not res.result_rows:
    print("No run_id found")
    exit()

run_id, source_path = res.result_rows[0]
print(f"Latest run_id: {run_id}")
print(f"Source path: {source_path}")

# 2. Fetch programs without run_id
print("Fetching programs without run_id...")
query = """
SELECT * 
FROM programs 
WHERE JSONExtractString(metadata, 'original_run_id') = ''
ORDER BY timestamp DESC
LIMIT 1000
"""
res = client.query(query)

if not res.result_rows:
    print("No programs without run_id found.")
    exit()

print(f"Found {len(res.result_rows)} programs.")

programs = []


class Program:
    def __init__(self, data, cols):
        for k, v in zip(cols, data):
            setattr(self, k, v)


for row in res.result_rows:
    programs.append(Program(row, res.column_names))

# 3. Update metadata string and timestamp
print("Updating metadata via INSERT...")
updated_rows = []
for i, program in enumerate(programs):
    meta_str = program.metadata

    # Simple string injection to avoid JSON parsing errors
    # Note: This assumes meta_str starts with '{'
    injection = f'"original_run_id": "{run_id}", "migration_source": "{source_path}", '

    if meta_str and meta_str.strip().startswith("{"):
        new_meta_str = meta_str.strip().replace("{", "{" + injection, 1)
    else:
        # Fallback if metadata is empty or broken
        new_meta_str = "{" + injection[:-2] + "}"  # Remove trailing comma

    program.metadata = new_meta_str
    program.timestamp = time.time()

    updated_rows.append(
        [
            program.id,
            program.code,
            program.language,
            program.parent_id,
            program.archive_inspiration_ids,
            program.top_k_inspiration_ids,
            program.generation,
            program.timestamp,
            program.code_diff,
            program.combined_score,
            program.public_metrics,
            program.private_metrics,
            program.text_feedback,
            program.complexity,
            program.embedding,
            program.embedding_pca_2d,
            program.embedding_pca_3d,
            program.embedding_cluster_id,
            1 if program.correct else 0,
            program.children_count,
            program.metadata,
            program.island_idx if program.island_idx is not None else -1,
            program.migration_history,
            1 if program.in_archive else 0,
        ]
    )

client.insert(
    "programs",
    updated_rows,
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
print(f"Updated {len(programs)} programs.")
