from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str
    age: Optional[int]
    email: Optional[EmailStr]
    cgpa: float = Field(gt=0.0, lt=4.0, description="cgpa should be between 0 and 4")
    
student = {
    "name": "John Doe",
    "age": '30',
    "email": "bL2dD@example.com",
    "cgpa": 3.9
}

user = Student(**student)
print(user)