import sys
import os
sys.path.insert(0, '.')
os.environ["PYTHONIOENCODING"] = "utf-8"
from main_server import smart_search_with_memory
import database as db

db.initialize_database()
print("Calling smart_search_with_memory(user_id=1, query='pan card')")
try:
    res = smart_search_with_memory(user_id=1, query="pan card")
    print("RESULT LENGTH:", len(res))
    if "RESULTS_JSON" in res:
        print("JSON found")
    else:
        print("NO JSON! Output was:")
        print(res)
except Exception as e:
    print("ERROR:", e)
