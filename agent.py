import os
import sys
from pathlib import Path
from openai import OpenAI

# ---------------- config ----------------
MODEL = "gpt-5"
MAX_FILE_CHARS = 8_000
INCLUDE_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".tf", ".tfvars"}
FILE_CONTEXT_TAG = "__FILE_CONTEXT__"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Conversation state (system + turns)
context: list[dict] = []

# Directory change tracking (for smart reload)
_last_sig = None


def project_dir() -> Path:
    """
    Directory to load files from:
    - If the user runs:   python path/to/agent.py ...
      then files should come from the *current working directory* (cwd),
      i.e., where they invoked the command from.
    """
    return Path.cwd().resolve()


def build_file_context_message(base_dir: Path) -> dict | None:
    """
    Build a single system message containing the contents of eligible files
    in `base_dir` (the directory the user invoked the script from).
    """
    included: list[str] = []
    file_blocks: list[str] = []

    # If agent.py happens to be in the same dir, skip it; otherwise this is harmless.
    agent_filename = Path(__file__).name

    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == agent_filename:
            continue
        if path.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
            included.append(path.name)
            file_blocks.append(f"File: {path.name}\n---\n{content}\n")
        except Exception as e:
            print(f"[warn] skipped {path.name}: {e}")

    print(f"[info] loaded {len(included)} files from {base_dir}: {included}")

    if not file_blocks:
        return None

    return {
        "role": "system",
        "content": (
            f"{FILE_CONTEXT_TAG}\n"
            f"Project directory: {base_dir}\n"
            "You have access to the following files from the project directory. "
            "Use them as context when answering. If the user asks about a file, "
            "prefer quoting the relevant snippet and explaining it.\n\n"
            + "\n".join(file_blocks)
        ),
    }


def add_initial_file_context():
    msg = build_file_context_message(project_dir())
    if msg:
        # Keep file context at the very front of the conversation
        context.insert(0, msg)


def reload_file_context():
    """
    Refresh file context without deleting the conversation history.
    """
    global context
    # Remove prior file-context system messages (only those we added)
    context = [
        m
        for m in context
        if not (m.get("role") == "system" and str(m.get("content", "")).startswith(FILE_CONTEXT_TAG))
    ]
    # Add updated file context back to the front (from cwd)
    add_initial_file_context()


def directory_signature(base_dir: Path) -> tuple:
    agent_filename = Path(__file__).name

    sig = []
    for p in sorted(base_dir.iterdir()):
        if p.is_file() and p.name != agent_filename and p.suffix.lower() in INCLUDE_EXTENSIONS:
            try:
                st = p.stat()
                sig.append((p.name, st.st_mtime_ns, st.st_size))
            except Exception:
                pass
    return tuple(sig)


def call():
    return client.responses.create(model=MODEL, input=context)


def process(prompt: str) -> str:
    global _last_sig

    base_dir = project_dir()
    sig = directory_signature(base_dir)
    if sig != _last_sig:
        reload_file_context()
        _last_sig = sig

    context.append({"role": "user", "content": prompt})
    response = call()
    output = response.output_text
    context.append({"role": "assistant", "content": output})
    return output


def main():
    add_initial_file_context()

    # One-shot CLI usage:
    #   python /path/to/agent.py "your prompt here"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(process(prompt))
        return

    # Interactive REPL usage:
    #   /reload  -> refresh directory files without losing conversation turns
    #   /exit    -> quit
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            if line in {"/exit", "/quit"}:
                print("Exiting.")
                break

            if line == "/reload":
                reload_file_context()
                print(">>> file context reloaded\n")
                continue

            result = process(line)
            print(f">>> {result}\n")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
