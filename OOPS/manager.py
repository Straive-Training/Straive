from employee import Employee
from bank_account import BankAccount

class Manager(Employee):
    def approve_loan(self, customer_account: BankAccount, amount: float) -> bool:
        if not customer_account.is_active:
            print("Cannot approve loan: Customer account is inactive.")
            return False
        max_loan = customer_account.balance * 2
        if amount <= max_loan:
            print(f"Loan of {amount:.2f} approved for account {customer_account.account_number}.")
            customer_account.deposit(amount)
            return True
        else:
            print(f"Loan amount exceeds limit of {max_loan:.2f}.")
            return False