from __future__ import annotations
from fastmcp import FastMCP

mcp = FastMCP(name="Arithmetic Math Calculator")

def _as_number(x):
    """Converts x to a number"""
    
    if isinstance(x, (int, float)):
        return float(x)
    
    if isinstance(x, str):
            return float(x.strip())
    return TypeError("Expected a number (int/float or number as string)")

@mcp.tool
def add_numbers(a: float, b: float) -> int:
    """Add two numbers."""
    return _as_number(a) + _as_number(b)

@mcp.tool
def multiply_numbers(a: float, b: float) -> int:
    """Multiply two numbers."""
    return _as_number(a) * _as_number(b)

@mcp.tool
def subtract_numbers(a: float, b: float) -> int:
    """Subtract two numbers."""
    return _as_number(a) - _as_number(b)

@mcp.tool
def divide_numbers(a: float, b: float) -> int:
    """Divide two numbers."""
    return _as_number(a) / _as_number(b) if b > 0 else TypeError("Cannot divide by zero")

@mcp.tool
def power(a: float) -> int:
    """Square a number."""
    return _as_number(a) ** 2

if __name__ == "__main__":
    mcp.run()