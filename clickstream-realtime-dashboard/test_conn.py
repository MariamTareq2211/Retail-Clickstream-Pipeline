import struct
import pyodbc
from azure.identity import InteractiveBrowserCredential

SERVER = "oe3xc2subvwevo757s5gzbossa-nby64cvvc6benanjye2rzyyu34.datawarehouse.fabric.microsoft.com"
DATABASE = "Retail_Clickstream_Lakehouse"
USER = "2401526@rst.edu.eg"

credential = InteractiveBrowserCredential(login_hint=USER)
token = credential.get_token("https://database.windows.net/.default").token

token_bytes = token.encode("utf-16-le")
token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={SERVER},1433;"
    f"Database={DATABASE};"
    "Encrypt=yes;TrustServerCertificate=no;",
    attrs_before={1256: token_struct},
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM Silver.stream_events")
print("stream_events rows:", cur.fetchone()[0])
conn.close()