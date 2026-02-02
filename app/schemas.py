from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        orm_mode = True

# Add this new schema for login response
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse