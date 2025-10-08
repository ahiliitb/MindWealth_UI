"""
Helper utility functions
"""


def find_column_by_keywords(columns, keywords):
    """Find a column name that contains any of the keywords"""
    for col in columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None

