import os
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

# 2. Get programs without original_run_id in metadata
# We assume they are recent programs
query = """
SELECT id, metadata 
FROM programs 
WHERE JSONExtractString(metadata, 'original_run_id') = ''
ORDER BY timestamp DESC
LIMIT 1000
"""

programs = client.query(query)
print(f"Found {len(programs.result_rows)} programs without run_id")

count = 0
for row in programs.result_rows:
    pid, meta_str = row
    try:
        meta = None
        try:
            # strict=False allows control characters like newlines in strings
            meta = json.loads(meta_str, strict=False)
        except json.JSONDecodeError:
            try:
                # Fallback: try to fix newlines manually if strict=False fails
                fixed_str = (
                    meta_str.replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                meta = json.loads(fixed_str, strict=False)
            except:
                pass

        if meta is None:
            print(f"Skipping {pid}: Could not parse metadata even with strict=False")
            continue

        # Check if we need to update
        if "original_run_id" not in meta or meta.get("original_run_id") != run_id:
            meta["original_run_id"] = run_id
            meta["migration_source"] = source_path

            # Update DB using parameters
            new_meta_str = json.dumps(meta)

            client.command(
                "ALTER TABLE programs UPDATE metadata = {meta:String} WHERE id = {pid:String}",
                parameters={"meta": new_meta_str, "pid": pid},
            )
            count += 1
    except Exception as e:
        print(f"Error updating {pid}: {e}")

print(f"Updated {count} programs")
