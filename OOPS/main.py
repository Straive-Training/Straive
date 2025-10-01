from bank_account import BankAccount
from manager import Manager
from current import CurrentAccount
from savings import SavingsAccount

class BankSystem:
    def __init__(self):
        self.__accounts = {}

    def add_account(self, account: BankAccount):
        self.__accounts[account.account_number] = account

    def find_account(self, account_number: int) -> BankAccount:
        return self.__accounts.get(account_number)

    def create_account(self, account_type: str, holder_name: str, initial_deposit: float):
        if account_type.lower() == "savings":
            account = SavingsAccount(holder_name, initial_deposit)
        elif account_type.lower() == "current":
            account = CurrentAccount(holder_name, initial_deposit)
        else:
            raise ValueError("Invalid account type.")
        self.add_account(account)
        print(f"Created {account_type.capitalize()} Account with Account Number: {account.account_number}")
        return account

    def display_all_accounts(self):
        print("\n--- All Bank Accounts ---")
        for acc in self.__accounts.values():
            print(acc)
            print("-" * 30)


def menu():
    bank = BankSystem()
    manager = Manager("John Smith", 1)  # Example manager

    while True:
        print("""
--- Bank Menu ---
1. Create new account
2. Deposit money
3. Withdraw money
4. Transfer money
5. View account details
6. Close account
7. Add interest to all savings accounts
8. Approve loan (Manager only)
9. Exit
""")
        choice = input("Enter your choice: ")

        if choice == "1":
            acc_type = input("Enter account type (savings/current): ").strip().lower()
            name = input("Account holder name: ").strip()
            try:
                initial_deposit = float(input("Initial deposit amount: "))
            except ValueError:
                print("Invalid amount.")
                continue
            try:
                bank.create_account(acc_type, name, initial_deposit)
            except ValueError as e:
                print(e)

        elif choice == "2":
            try:
                acc_num = int(input("Account number: "))
                amount = float(input("Amount to deposit: "))
            except ValueError:
                print("Invalid input.")
                continue
            account = bank.find_account(acc_num)
            if account:
                account.deposit(amount)
            else:
                print("Account not found.")

        elif choice == "3":
            try:
                acc_num = int(input("Account number: "))
                amount = float(input("Amount to withdraw: "))
            except ValueError:
                print("Invalid input.")
                continue
            account = bank.find_account(acc_num)
            if account:
                account.withdraw(amount)
            else:
                print("Account not found.")

        elif choice == "4":
            try:
                acc_num = int(input("Your account number: "))
                target_acc_num = int(input("Target account number: "))
                amount = float(input("Amount to transfer: "))
            except ValueError:
                print("Invalid input.")
                continue
            source_account = bank.find_account(acc_num)
            target_account = bank.find_account(target_acc_num)
            if source_account and target_account:
                source_account.transfer(amount, target_account)
            else:
                print("One or both accounts not found.")

        elif choice == "5":
            try:
                acc_num = int(input("Account number: "))
            except ValueError:
                print("Invalid input.")
                continue
            account = bank.find_account(acc_num)
            if account:
                print(account)
                account.display_transaction_history()
            else:
                print("Account not found.")

        elif choice == "6":
            try:
                acc_num = int(input("Account number to close: "))
            except ValueError:
                print("Invalid input.")
                continue
            account = bank.find_account(acc_num)
            if account:
                account.close_account()
            else:
                print("Account not found.")

        elif choice == "7":
            # Add interest to all savings accounts
            for acc in bank._BankSystem__accounts.values():
                if isinstance(acc, SavingsAccount):
                    acc.add_interest()

        elif choice == "8":
            # Manager approves loan
            try:
                acc_num = int(input("Customer account number for loan approval: "))
                amount = float(input("Loan amount: "))
            except ValueError:
                print("Invalid input.")
                continue
            customer_acc = bank.find_account(acc_num)
            if customer_acc:
                manager.approve_loan(customer_acc, amount)
            else:
                print("Customer account not found.")

        elif choice == "9":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select from the menu.")

if __name__ == "__main__":
    menu()
