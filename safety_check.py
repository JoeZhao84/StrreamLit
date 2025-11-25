# =============================
# Safety filter
# =============================
def is_code_safe(code: str) -> bool:
    """
    Naive safety checks to avoid obviously dangerous code.
    This is NOT bullet-proof; consider using AST-based sandboxing in production.
    """
    forbidden = [
        "import ",
        "open(",
        "exec(",
        "eval(",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "pickle",
        "shutil",
        "Popen",
        "input(",
    ]
    lowered = code.lower()
    return not any(token in lowered for token in forbidden)


