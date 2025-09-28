class UserNotFound(Exception):
    """
    Exception raised when a user is not found in the database.
    """

    pass


class PasswordDoesNotMatch(Exception):
    """
    Exception raised when a provided password does not match the stored hash.
    """

    pass


class UserAlreadyExists(Exception):
    """
    Exception raised when an attempt is made to create a user with an
    email that already exists.
    """

    pass


class UserIsNotActive(Exception):
    """
    Exception raised when an operation is attempted on a user account that is not active.
    """

    pass
