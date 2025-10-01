from bank_account import BankAccount

class SavingsAccount(BankAccount):
    def __init__(self, account_holder: str, balance: float = 0, interest_rate: float = 0.04):
        super().__init__(account_holder, balance)
        self.__interest_rate = interest_rate

    @property
    def interest_rate(self):
        return self.__interest_rate

    def add_interest(self):
        if not self.is_active:
            print("Account inactive; cannot add interest.")
            return False
        interest = self.balance * self.__interest_rate
        self.deposit(interest)
        print(f"Interest of {interest:.2f} added to account {self.account_number}.")
        return True

    def __str__(self):
        base_str = super().__str__()
        return base_str + f"\nAccount Type: Savings\nInterest Rate: {self.__interest_rate * 100:.2f}%"

