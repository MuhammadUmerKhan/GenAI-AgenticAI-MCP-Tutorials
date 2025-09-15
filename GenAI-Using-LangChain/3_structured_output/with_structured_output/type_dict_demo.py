from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

user: User = {
    "name": "John Doe",
    "age": 30
}

print(user)