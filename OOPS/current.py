from bank_account import BankAccount

class CurrentAccount(BankAccount):
    def __init__(self, account_holder: str, balance: float = 0, overdraft_limit: float = 500):
        super().__init__(account_holder, balance)
        self.__overdraft_limit = overdraft_limit

    @property
    def overdraft_limit(self):
        return self.__overdraft_limit

    def withdraw(self, amount: float):
        if not self.is_active:
            print("Account inactive. Withdrawal failed.")
            return False
        if amount <= 0:
            print("Withdrawal amount must be positive.")
            return False
        if amount > self.balance + self.__overdraft_limit:
            print(f"Exceeded overdraft limit! Max available: {self.balance + self.__overdraft_limit:.2f}")
            return False
        # Overdraft allowed, reduce balance possibly below zero
        self._BankAccount__balance = self.balance - amount
        self._BankAccount__log_transaction("Withdrawal", amount)
        print(f"Withdrawn {amount:.2f}. New balance: {self.balance:.2f} (includes overdraft)")
        return True

    def __str__(self):
        base_str = super().__str__()
        return base_str + f"\nAccount Type: Current\nOverdraft Limit: {self.__overdraft_limit:.2f}"
