"""
Configuration for Advanced CrewAI Demo
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory if not found locally
local_env = Path(__file__).parent / '.env'
parent_env = Path(__file__).parent.parent / '.env'

if local_env.exists():
    load_dotenv(local_env)
elif parent_env.exists():
    load_dotenv(parent_env)
else:
    load_dotenv()


class Config:
    """Configuration class for API keys and settings"""

    # OpenAI API Key (required)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Brave Search API Key (required for search)
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # MCP settings
    ENABLE_MCP_THINKING = os.getenv("ENABLE_MCP_THINKING", "true").lower() == "true"

    # Search settings
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "3"))

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file or environment variables.\n"
                "Example: export OPENAI_API_KEY='your-api-key-here'"
            )

        if not cls.BRAVE_API_KEY:
            raise ValueError(
                "BRAVE_API_KEY not found. Please set it in .env file or environment variables.\n"
                "Example: export BRAVE_API_KEY='your-brave-api-key-here'\n"
                "Get your key at: https://brave.com/search/api/"
            )

        # Set the API keys in environment for libraries to use
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        os.environ["BRAVE_API_KEY"] = cls.BRAVE_API_KEY

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("🔧 Configuration:")
        print(f"  Model: {cls.DEFAULT_MODEL}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  MCP Sequential Thinking: {'✓ Enabled' if cls.ENABLE_MCP_THINKING else '✗ Disabled'}")
        print(f"  OpenAI API Key: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not set'}")
        print(f"  Brave Search API Key: {'✓ Set' if cls.BRAVE_API_KEY else '✗ Not set'}")
        print()


if __name__ == "__main__":
    # Test configuration
    try:
        Config.validate()
        Config.print_config()
        print("✅ Configuration is valid!")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
