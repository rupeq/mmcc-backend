class IdColumnRequiredException(Exception):
    """Exception raised when the 'id' column is required but not provided."""

    pass


class SimulationNotFound(Exception):
    """Exception raised when the simulation is not found."""

    pass


class SimulationReportNotFound(Exception):
    """Exception raised when the report is not found."""

    pass


class SimulationReportsNotFound(Exception):
    """Exception raised when the reports are not found."""

    pass


class BackgroundTaskNotFound(Exception):
    """Exception raised when the background task is not found."""

    pass
