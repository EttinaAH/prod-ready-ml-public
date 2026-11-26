from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return "Hello, world!"

@app.get("/greet/{name}")
def greet(name: str):
    return f"Hello, {name}!"

@app.get("/greet_int")
def greet_int(num: int = 1):
    return f"Hello, number {num}!"

@app.get("/greet_int_name/{name}")
def greet_int_name(name: str, num: int = 1):
    return f"Hello, {name}! Your number is {num}."