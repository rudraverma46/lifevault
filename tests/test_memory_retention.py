import os
import shutil
import time
from pathlib import Path

# Add project root to sys.path implicitly for testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lifevault import (
    VAULT_DIR, 
    fast_query, 
    clear_memory, 
    ingest_vault, 
    save_chat_memory,
    collection,
    startup_health_check
)

def wait_for_async_save():
    """Wait briefly for async memory saving threads to flush to DB."""
    time.sleep(1.0)

def extract_answer(gen):
    """Exhaust generator and return the final text answer."""
    ans = ""
    for update in gen:
        ans = update.get("answer", "")
    return ans

def test_refresh_simulation():
    """Prove that a fact survives a hard context drop (history=[])."""
    print("\n--- TEST: Refresh Simulation ---")
    question_1 = "My favorite color is neon purple."
    ans_1 = extract_answer(fast_query(question_1, history=[]))
    wait_for_async_save()
    
    # Simulate page refresh by passing empty history.
    question_2 = "What is my favorite color?"
    ans_2 = extract_answer(fast_query(question_2, history=[]))
    
    print(f"Fact Injected: {question_1}")
    print(f"Recall Check: {ans_2}")
    
    if "neon purple" in ans_2.lower():
        print("✅ Pass: Context survived history drop.")
    else:
        print("❌ Fail: Fact forgotten without history.")

def test_rebuild_safety():
    """Prove that ingest_vault successfully indexes chat_history.md."""
    print("\n--- TEST: Rebuild Safety ---")
    
    # Check current memory count
    mem_1 = collection.get(where={"type": "memory"})
    count_1 = len(mem_1.get("ids", []))
    
    # Destroy ChromaDB instance mimicking a system wipe or script removal
    # We won't literally delete the DB from disk since Chroma is holding a lock.
    # Instead, we delete all memory nodes to mathematically simulate the loss.
    collection.delete(where={"type": "memory"})
    
    mem_2 = collection.get(where={"type": "memory"})
    count_2 = len(mem_2.get("ids", []))
    print(f"Memories before wipe: {count_1} | After specific wipe: {count_2}")
    
    # Re-ingest
    for _, _ in ingest_vault():
        pass
        
    mem_3 = collection.get(where={"type": "memory"})
    count_3 = len(mem_3.get("ids", []))
    print(f"Memories after ingest_vault(): {count_3}")
    
    if count_3 > 0:
        print("✅ Pass: chat_history.md properly indexed during rebuild.")
    else:
        print("❌ Fail: chat_history.md was ignored by ingest_vault.")

def test_conflict_resolution():
    """Prove that newer injected facts cleanly override older facts."""
    print("\n--- TEST: Conflict Resolution ---")
    question_1 = "I adopted a cat named Whiskers."
    extract_answer(fast_query(question_1, history=[]))
    wait_for_async_save()
    
    question_2 = "Actually, I renamed my cat to Shadow."
    extract_answer(fast_query(question_2, history=[]))
    wait_for_async_save()
    
    question_3 = "What is the name of my cat?"
    ans_3 = extract_answer(fast_query(question_3, history=[]))
    
    print(f"Initial Name: Whiskers | Updated Name: Shadow")
    print(f"Recall Check: {ans_3}")
    
    if "shadow" in ans_3.lower() and "whiskers" not in ans_3.lower():
        print("✅ Pass: Conflicting fact successfully resolved to newer state.")
    else:
        print("⚠️ Warning/Fail: LLM struggled with conflicting timeline facts.")

if __name__ == "__main__":
    health = startup_health_check()
    if not health["ok"]:
        print(f"Health fail: {health['message']}")
        sys.exit(1)
        
    print("Beginning validation test suite...")
    test_refresh_simulation()
    test_rebuild_safety()
    test_conflict_resolution()
    print("\nTests complete.")
