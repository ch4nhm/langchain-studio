.PHONY: ingest init start test

init:
	cd backend && poetry install
	cd frontend && npm install

ingest:
	@echo "Generating dummy data if not exists..."
	cd backend && poetry run python scripts/generate_dummy_data.py
	@echo "Running ETL pipeline..."
	cd backend && poetry run python scripts/ingest.py

start-backend:
	cd backend && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

start-frontend:
	cd frontend && npm run dev

test:
	cd backend && poetry run pytest --cov=app tests/

lint:
	cd backend && poetry run black app tests
	cd backend && poetry run isort app tests
	cd backend && poetry run flake8 app tests
	cd backend && poetry run mypy app
