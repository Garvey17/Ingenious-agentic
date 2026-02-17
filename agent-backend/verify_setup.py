"""
Setup verification script for Phase 1.
Run this after installing dependencies to verify the setup.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 encoding for stdout/stderr
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def verify_imports():
    """Verify all required packages can be imported."""
    print("[*] Verifying package imports...")
    
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("pydantic_settings", "Pydantic Settings"),
        ("openai", "OpenAI SDK"),
        ("google.generativeai", "Google Generative AI"),
        ("langgraph", "LangGraph"),
        ("qdrant_client", "Qdrant Client"),
        ("httpx", "HTTPX"),
        ("aiohttp", "aiohttp"),
        ("dotenv", "python-dotenv"),
    ]
    
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0


def verify_structure():
    """Verify project structure."""
    print("\n[*] Verifying project structure...")
    
    base_path = Path(__file__).parent
    
    required_paths = [
        "app/__init__.py",
        "app/main.py",
        "app/config/settings.py",
        "app/config/logging.py",
        "app/models/llm.py",
        "app/models/embeddings.py",
        "app/agents/base_agent.py",
        "app/graph/state.py",
        "app/graph/builder.py",
        "app/prompts/system_prompts.py",
        "app/prompts/schemas.py",
        "app/tools/base.py",
        "app/mcp/client.py",
        "app/memory/qdrant_client.py",
        "app/api/research.py",
        ".env.example",
        "requirements.txt",
        "README.md",
    ]
    
    missing = []
    for path_str in required_paths:
        path = base_path / path_str
        if path.exists():
            print(f"  [OK] {path_str}")
        else:
            print(f"  [FAIL] {path_str} (missing)")
            missing.append(path_str)
    
    return len(missing) == 0


def verify_config():
    """Verify configuration can be loaded."""
    print("\n[*] Verifying configuration...")
    
    try:
        from app.config.settings import settings
        
        print(f"  [OK] Settings loaded successfully")
        print(f"     App: {settings.app_name} v{settings.app_version}")
        print(f"     Environment: {settings.environment}")
        print(f"     LLM Provider: {settings.llm_provider}")
        print(f"     Search Provider: {settings.search_provider}")
        
        # Check for API keys (without printing them)
        if settings.llm_provider == "openai" and not settings.openai_api_key:
            print(f"  [WARN] OPENAI_API_KEY not set")
        elif settings.llm_provider == "gemini" and not settings.google_api_key:
            print(f"  [WARN] GOOGLE_API_KEY not set")
        
        if settings.search_provider == "tavily" and not settings.tavily_api_key:
            print(f"  [WARN] TAVILY_API_KEY not set")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Configuration error: {e}")
        return False


def verify_llm():
    """Verify LLM can be initialized."""
    print("\n[*] Verifying LLM initialization...")
    
    try:
        from app.models.llm import get_llm
        from app.config.settings import settings
        
        # Only try to initialize if API key is set
        if settings.llm_provider == "openai" and not settings.openai_api_key:
            print(f"  [WARN] Skipping (OPENAI_API_KEY not set)")
            return True
        elif settings.llm_provider == "gemini" and not settings.google_api_key:
            print(f"  [WARN] Skipping (GOOGLE_API_KEY not set)")
            return True
        
        llm = get_llm()
        print(f"  [OK] LLM initialized: {llm.__class__.__name__}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] LLM initialization error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print(" PHASE 1 SETUP VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Package Imports", verify_imports),
        ("Project Structure", verify_structure),
        ("Configuration", verify_config),
        ("LLM Initialization", verify_llm),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print(" VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[OK] All checks passed! Phase 1 setup is complete.")
        print("\nNext steps:")
        print("   1. Ensure .env file has your API keys")
        print("   2. Run: uvicorn app.main:app --reload")
        print("   3. Test: curl http://localhost:8000/health")
        print("   4. Proceed to Phase 2: Tool System Implementation")
        return 0
    else:
        print("\n[FAIL] Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("   - Run: pip install -r requirements.txt")
        print("   - Copy .env.example to .env and add API keys")
        return 1


if __name__ == "__main__":
    sys.exit(main())
