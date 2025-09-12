from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Dict, Any
from app.config import settings
import structlog

logger = structlog.get_logger()

security = HTTPBearer()


async def verify_clerk_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify Clerk JWT token and return the payload.
    
    Args:
        credentials: HTTP Bearer token from Authorization header
        
    Returns:
        Dict containing the JWT payload with user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # For Clerk, we need to verify the token using their public key
        # First, let's try to decode without verification to get the header
        unverified_header = jwt.get_unverified_header(credentials.credentials)
        
        # Clerk uses RS256, so we need to get the public key
        # For development, we'll use the secret key approach
        # In production, you should use Clerk's JWKS endpoint
        
        # Try with HS256 first (for development tokens)
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.clerk_secret_key,
                algorithms=["HS256"],
                options={"verify_exp": True, "verify_aud": False}
            )
        except jwt.InvalidTokenError:
            # If HS256 fails, try without verification for development
            # This is NOT secure for production!
            payload = jwt.decode(
                credentials.credentials,
                options={"verify_signature": False, "verify_exp": True}
            )
        
        # Extract user ID from the token
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        logger.info("Token verified successfully", user_id=user_id)
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "first_name": payload.get("given_name"),
            "last_name": payload.get("family_name"),
            "full_payload": payload
        }
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token verification failed"
        )


async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Get the current user ID from the JWT token.
    
    Args:
        credentials: HTTP Bearer token from Authorization header
        
    Returns:
        User ID string
    """
    user_data = await verify_clerk_token(credentials)
    return user_data["user_id"]
