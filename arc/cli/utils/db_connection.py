"""Database connection manager for Arc CLI."""

import os

from arc.database.client import ArcDBClient
from arc.cli.utils.console import ArcConsole

console = ArcConsole()


class DatabaseConnectionManager:
    """Manages database connections for the Arc CLI with fallback support."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self._client: ArcDBClient | None = None
        self._is_connected = False
        self._connection_url = os.environ.get("TIMESCALE_SERVICE_URL")
    
    @property
    def is_available(self) -> bool:
        """Check if database connection is available."""
        return self._is_connected
    
    async def initialize(self) -> bool:
        """Initialize database connection with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self._connection_url:
            console.print(
                "Database connection not configured. Running in file-only mode.",
                style="warning"
            )
            return False
            
        try:
            self._client = ArcDBClient(connection_string=self._connection_url)
            await self._client.initialize()
            self._is_connected = True
            return True
        except Exception as e:
            console.print(
                f"Database connection failed: {str(e)}. Running in file-only mode.",
                style="warning"
            )
            self._is_connected = False
            return False
    
    async def close(self):
        """Close database connection if open."""
        if self._client:
            await self._client.close()
            self._is_connected = False
    
    def get_client(self):
        """Get database client if available.
        
        Returns:
            ArcDBClient if connected, None otherwise
        """
        if self._is_connected and self._client:
            return self._client
        else:
            return None
    
    async def health_check(self) -> bool:
        """Perform health check on database connection.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._is_connected or not self._client:
            return False
            
        try:
            # Simple query to check connection
            await self._client.engine.execute("SELECT 1")
            return True
        except Exception:
            self._is_connected = False
            return False


# Global instance for CLI commands
db_manager = DatabaseConnectionManager()