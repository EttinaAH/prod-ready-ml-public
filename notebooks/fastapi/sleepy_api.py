from fastapi import FastAPI
import asyncio
import time

app = FastAPI()

@app.get("/sleepio/{seconds}")
async def sleep_for(seconds: int):
    start = time.time()

    print("Going to bed")
    for s in range(seconds+1):
      await asyncio.sleep(1)
      print(f"zz {s}")

    end = time.time()
    print(f"Total sleep time: {end-start} seconds")

    return "Awake!"
