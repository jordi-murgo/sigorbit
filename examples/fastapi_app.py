"""Run with: uvicorn examples.fastapi_app:app --reload"""

from sigorbit.api import create_app

app = create_app()
