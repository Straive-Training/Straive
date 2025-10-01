class Employee:
    def __init__(self, name: str, employee_id: int):
        self.__name = name
        self.__employee_id = employee_id

    @property
    def name(self):
        return self.__name

    @property
    def employee_id(self):
        return self.__employee_id

    def __str__(self):
        return f"Employee {self.__name} (ID: {self.__employee_id})"