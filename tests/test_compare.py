import pytest
from app.models.user import User
from app.models.swing import Swing


def test_compare_swings_success(client, db_session, sample_user_data):
    """Test successful swing comparison."""
    # Create user
    user = User(**sample_user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Create two swings
    swing1 = Swing(
        user_id=user.id,
        file_url="s3://test-bucket/swing1.mp4",
        tag="slice"
    )
    swing2 = Swing(
        user_id=user.id,
        file_url="s3://test-bucket/swing2.mp4",
        tag="hook"
    )
    db_session.add(swing1)
    db_session.add(swing2)
    db_session.commit()
    db_session.refresh(swing1)
    db_session.refresh(swing2)
    
    # Compare swings
    response = client.get(f"/compare/{swing1.id}/{swing2.id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "swing1" in data
    assert "swing2" in data
    assert data["swing1"]["id"] == swing1.id
    assert data["swing2"]["id"] == swing2.id
    assert data["swing1"]["fileUrl"] == swing1.file_url
    assert data["swing2"]["fileUrl"] == swing2.file_url


def test_compare_swings_not_found(client, db_session, sample_user_data):
    """Test comparison with non-existent swings."""
    # Create user
    user = User(**sample_user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Try to compare non-existent swings
    response = client.get("/compare/999/1000")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"]


def test_compare_swings_access_denied(client, db_session, sample_user_data):
    """Test comparison of swings from different users."""
    # Create two users
    user1 = User(**sample_user_data)
    user2 = User(
        id="test_user_456",
        email="jane@example.com",
        first_name="Jane",
        last_name="Doe"
    )
    db_session.add(user1)
    db_session.add(user2)
    db_session.commit()
    db_session.refresh(user1)
    db_session.refresh(user2)
    
    # Create swings for different users
    swing1 = Swing(
        user_id=user1.id,
        file_url="s3://test-bucket/swing1.mp4",
        tag="slice"
    )
    swing2 = Swing(
        user_id=user2.id,
        file_url="s3://test-bucket/swing2.mp4",
        tag="hook"
    )
    db_session.add(swing1)
    db_session.add(swing2)
    db_session.commit()
    db_session.refresh(swing1)
    db_session.refresh(swing2)
    
    # Try to compare swings from different users
    response = client.get(f"/compare/{swing1.id}/{swing2.id}")
    
    assert response.status_code == 403
    data = response.json()
    assert "Access denied" in data["detail"]
