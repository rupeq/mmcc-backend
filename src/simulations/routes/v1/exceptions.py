class BadFilterFormat(Exception):
    """Exception raised for errors in the input filter format."""

    pass


class InvalidColumn(Exception):
    """Exception raised for errors in column names."""

    pass


class InvalidReportStatus(Exception):
    """Exception raised for errors in report status."""

    pass
