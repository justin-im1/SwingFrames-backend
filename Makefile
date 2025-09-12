.PHONY: help install dev test clean docker-up docker-down migrate init-db

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Run development server
	python run_dev.py

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=app --cov-report=html

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MESSAGE="description")
	alembic revision --autogenerate -m "$(MESSAGE)"

init-db: ## Initialize database with sample data
	python init_db.py

setup: install ## Complete setup for development
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp env.example .env; echo "Created .env file from template. Please update with your configuration."; fi
	@echo "Setup complete! Run 'make dev' to start the development server."

lint: ## Run linting
	flake8 app tests
	black --check app tests

format: ## Format code
	black app tests
	isort app tests
