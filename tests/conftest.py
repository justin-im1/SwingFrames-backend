import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
from app.main import app
from app.database import get_db, Base
from app.models import user, swing, comparison
from app.middleware.auth import get_current_user_id

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def mock_get_current_user_id():
    return "test_user_123"


@pytest.fixture(scope="function")
def db_session():
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = TestingSessionLocal()
    
    yield session
    
    # Clean up
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_id] = mock_get_current_user_id
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_user_data():
    return {
        "id": "test_user_123",
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe"
    }


@pytest.fixture
def sample_swing_data():
    return {
        "fileUrl": "s3://test-bucket/swing1.mp4",
        "tag": "slice"
    }
