import os
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

query = """
SELECT 
    JSONExtractString(metadata, 'original_run_id') as run_id,
    JSONExtractString(metadata, 'migration_source') as source_path,
    count() as total,
    sum(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as working,
    max(timestamp) as last_ts
FROM programs
WHERE JSONExtractString(metadata, 'original_run_id') != ''
GROUP BY run_id, source_path
ORDER BY last_ts DESC
LIMIT 5
"""

res = client.query(query)
for row in res.result_rows:
    print(row)
