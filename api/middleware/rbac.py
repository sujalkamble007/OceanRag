"""
api/middleware/rbac.py — Role-Based Access Control definitions and helpers.
"""

ROLE_PERMISSIONS = {
    "admin": {
        "can_query": True,
        "can_view_history": True,
        "can_view_all_users": True,
        "can_view_experiments": True,
        "can_delete": True,
        "allowed_llms": "all",
        "allowed_retrievers": "all",
        "max_top_k": 10,
        "response_complexity": "expert",
    },
    "researcher": {
        "can_query": True,
        "can_view_history": True,
        "can_view_all_users": False,
        "can_view_experiments": True,
        "can_delete": False,
        "allowed_llms": "all",
        "allowed_retrievers": "all",
        "max_top_k": 10,
        "response_complexity": "detailed",
    },
    "student": {
        "can_query": True,
        "can_view_history": True,
        "can_view_all_users": False,
        "can_view_experiments": True,
        "can_delete": False,
        "allowed_llms": ["groq-llama8b", "groq-llama70b", "zephyr-7b"],
        "allowed_retrievers": ["similarity", "mmr"],
        "max_top_k": 5,
        "response_complexity": "simple",
    },
    "common_user": {
        "can_query": True,
        "can_view_history": True,
        "can_view_all_users": False,
        "can_view_experiments": False,
        "can_delete": False,
        "allowed_llms": ["groq-llama8b"],
        "allowed_retrievers": ["similarity"],
        "max_top_k": 3,
        "response_complexity": "simple",
    },
}


def check_permission(role: str, permission: str) -> bool:
    """Returns True if the role has the given permission."""
    return ROLE_PERMISSIONS.get(role, {}).get(permission, False)


def get_role_config(role: str) -> dict:
    """Returns the full permission config for a role."""
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["common_user"])
