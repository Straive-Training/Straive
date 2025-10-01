import itertools
from datetime import datetime

class BankAccount:
    _account_number_generator = itertools.count(1001)  # Unique account numbers

    def __init__(self, account_holder: str, balance: float = 0):
        self.__account_holder = account_holder
        self.__balance = balance
        self.__account_number = next(BankAccount._account_number_generator)
        self.__active = True
        self.__transaction_history = []
        self.__log_transaction("Account opened", 0)

    # Encapsulation: getters only
    @property
    def account_holder(self):
        return self.__account_holder

    @property
    def account_number(self):
        return self.__account_number

    @property
    def balance(self):
        return self.__balance

    @property
    def is_active(self):
        return self.__active

    def __log_transaction(self, transaction_type: str, amount: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__transaction_history.append({
            "timestamp": timestamp,
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.__balance,
        })

    def deposit(self, amount: float):
        if not self.__active:
            print("Account inactive. Deposit failed.")
            return False
        if amount <= 0:
            print("Deposit amount must be positive.")
            return False
        self.__balance += amount
        self.__log_transaction("Deposit", amount)
        print(f"Deposited {amount:.2f}. New balance: {self.__balance:.2f}")
        return True

    def withdraw(self, amount: float):
        if not self.__active:
            print("Account inactive. Withdrawal failed.")
            return False
        if amount <= 0:
            print("Withdrawal amount must be positive.")
            return False
        if amount > self.__balance:
            print("Insufficient funds!")
            return False
        self.__balance -= amount
        self.__log_transaction("Withdrawal", amount)
        print(f"Withdrawn {amount:.2f}. New balance: {self.__balance:.2f}")
        return True

    def transfer(self, amount: float, target_account: 'BankAccount'):
        if not self.__active:
            print("Your account is inactive. Transfer aborted.")
            return False
        if not target_account.is_active:
            print("Target account inactive. Transfer aborted.")
            return False
        if amount <= 0:
            print("Transfer amount must be positive.")
            return False
        if amount > self.__balance:
            print("Insufficient funds for transfer.")
            return False

        self.__balance -= amount
        self.__log_transaction(f"Transfer to {target_account.account_number}", amount)

        # Deposit to target account
        target_account.deposit(amount)
        # Log deposit manually to maintain consistency
        target_account._BankAccount__log_transaction(f"Transfer from {self.__account_number}", amount)

        print(f"Transferred {amount:.2f} to account {target_account.account_number}")
        return True

    def close_account(self):
        if not self.__active:
            print("Account already inactive.")
            return False
        self.__active = False
        self.__log_transaction("Account closed", 0)
        print(f"Account {self.__account_number} closed.")
        return True

    def get_transaction_history(self):
        return list(self.__transaction_history)  # Return copy

    def display_transaction_history(self):
        print(f"\nTransaction History for Account {self.__account_number}:")
        for tx in self.__transaction_history:
            print(f"{tx['timestamp']} | {tx['type']} | Amount: {tx['amount']:.2f} | Balance after: {tx['balance_after']:.2f}")

    def __str__(self):
        status = "Active" if self.__active else "Inactive"
        return (f"Account Number: {self.__account_number}\n"
                f"Holder: {self.__account_holder}\n"
                f"Balance: {self.__balance:.2f}\n"
                f"Status: {status}")