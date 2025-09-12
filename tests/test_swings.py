import pytest
from app.models.user import User
from app.models.swing import Swing


def test_create_swing_success(client, db_session, sample_swing_data):
    """Test successful swing creation."""
    # Create swing (user will be created automatically)
    response = client.post(
        "/swings",
        json=sample_swing_data
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "swingId" in data
    assert data["status"] == "saved"
    assert data["swingId"] > 0


def test_get_user_swings_success(client, db_session, sample_user_data, sample_swing_data):
    """Test retrieving user swings."""
    # Create user
    user = User(**sample_user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Create swing
    swing = Swing(
        user_id=user.id,
        file_url=sample_swing_data["fileUrl"],
        tag=sample_swing_data["tag"]
    )
    db_session.add(swing)
    db_session.commit()
    
    # Get swings
    response = client.get("/swings")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["file_url"] == sample_swing_data["fileUrl"]
    assert data[0]["tag"] == sample_swing_data["tag"]


def test_get_user_swings_empty(client, db_session):
    """Test retrieving swings for user with no swings."""
    # Get swings (no user or swings created)
    response = client.get("/swings")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0
