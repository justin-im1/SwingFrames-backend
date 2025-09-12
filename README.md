# Golf Swing Analyzer Backend

A FastAPI-based backend service for analyzing and comparing golf swings with video storage on AWS S3 and authentication via Clerk.

## Features

- **Video Upload**: Secure presigned URL generation for direct S3 uploads
- **Swing Management**: Store and retrieve golf swing metadata
- **Comparison**: Compare two swings from the same user
- **Authentication**: JWT-based authentication with Clerk
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Testing**: Comprehensive test suite with pytest
- **Docker**: Full containerization with docker-compose

## API Endpoints

### Upload (No Authentication Required)
- `POST /upload-url` - Generate presigned S3 URL for video upload

### Swings (Authentication Required)
- `POST /swings` - Save swing metadata
- `GET /swings` - Get all swings for the current user

### Comparison (Authentication Required)
- `GET /compare/{swingId1}/{swingId2}` - Compare two swings

### Authentication
All protected endpoints require a valid JWT token from Clerk in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- AWS account with S3 bucket
- Clerk account for authentication

### Environment Setup

1. Copy the environment template:
```bash
cp env.example .env
```

2. Update `.env` with your configuration:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/golf_swing_analyzer
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name
SECRET_KEY=your-secret-key-here
CLERK_SECRET_KEY=sk_test_your_clerk_secret_key
CLERK_PUBLISHABLE_KEY=pk_test_your_clerk_publishable_key
ENVIRONMENT=development
```

### Running with Docker Compose

1. Start the services:
```bash
docker-compose up -d
```

2. Run database migrations:
```bash
docker-compose exec app alembic upgrade head
```

3. The API will be available at `http://localhost:8000`

### Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start PostgreSQL (or use Docker):
```bash
docker run -d --name postgres -e POSTGRES_DB=golf_swing_analyzer -e POSTGRES_USER=user -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:15
```

3. Run migrations:
```bash
alembic upgrade head
```

4. Start the development server:
```bash
uvicorn app.main:app --reload
```

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app
```

## Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:
```bash
alembic upgrade head
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
├── app/
│   ├── models/          # SQLAlchemy database models
│   ├── schemas/         # Pydantic request/response schemas
│   ├── routers/         # FastAPI route handlers
│   ├── services/        # Business logic services
│   ├── database.py      # Database configuration
│   ├── config.py        # Application settings
│   └── main.py          # FastAPI application
├── alembic/             # Database migrations
├── tests/               # Test suite
├── docker-compose.yml   # Docker services
├── Dockerfile          # Application container
└── requirements.txt    # Python dependencies
```

## Deployment

The application is ready for deployment with:
- Gunicorn WSGI server
- Health check endpoints
- Structured logging
- Environment-based configuration

For production deployment, ensure:
1. Set proper CORS origins
2. Use environment variables for secrets
3. Configure proper database connection pooling
4. Set up monitoring and logging
5. Use HTTPS in production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request
