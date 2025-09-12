import pytest
from unittest.mock import patch, MagicMock


def test_get_upload_url_success(client):
    """Test successful presigned URL generation."""
    with patch('app.routers.upload.S3Service') as mock_s3_service:
        # Mock the S3 service
        mock_instance = MagicMock()
        mock_instance.generate_presigned_upload_url.return_value = (
            "https://test-bucket.s3.amazonaws.com/presigned-url",
            "s3://test-bucket/swings/test-file.mp4"
        )
        mock_s3_service.return_value = mock_instance
        
        response = client.post(
            "/upload-url",
            json={
                "filename": "swing1.mp4",
                "contentType": "video/mp4"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "uploadUrl" in data
        assert "fileUrl" in data
        assert data["fileUrl"].startswith("s3://")


def test_get_upload_url_invalid_request(client):
    """Test upload URL request with invalid data."""
    response = client.post(
        "/upload-url",
        json={
            "filename": "swing1.mp4"
            # Missing contentType
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_get_upload_url_s3_error(client):
    """Test upload URL generation when S3 service fails."""
    with patch('app.routers.upload.S3Service') as mock_s3_service:
        # Mock S3 service to raise an exception
        mock_instance = MagicMock()
        mock_instance.generate_presigned_upload_url.side_effect = Exception("S3 error")
        mock_s3_service.return_value = mock_instance
        
        response = client.post(
            "/upload-url",
            json={
                "filename": "swing1.mp4",
                "contentType": "video/mp4"
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to generate upload URL" in data["detail"]
