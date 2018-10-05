from enum import Enum


class Priority(Enum):

    HIGHEST = 0
    VERY_HIGH = 20
    HIGH = 40
    NORMAL = 50
    LOW = 60
    VERY_LOW = 80
    LOWEST = 100


def get_priority(priority):
    """Get priority value.

    Args:
        priority (int or str or :obj:`Priority`): Priority.

    Returns:
        int: The priority value.
    """
    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError('priority must be between 0 and 100')
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')
