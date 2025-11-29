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

pid = "812f0ecd-e476-439c-8a02-b4359888388e"
res = client.query(f"SELECT metadata FROM programs WHERE id = '{pid}'")
print(res.result_rows)
