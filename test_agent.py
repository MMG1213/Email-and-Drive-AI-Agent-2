import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import asyncio
from main___app import SmartAgent
import database as db

db.initialize_database()

agent = SmartAgent(user_id=1, model_type="groq")
response3 = agent.chat("fetch my recent emails", [])
print("RESPONSE 3:", response3)
