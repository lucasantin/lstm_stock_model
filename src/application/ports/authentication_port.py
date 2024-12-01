
from application.ports.dtos.login_dto import LoginDto

class AuthenticationPort:
  def authenticate(self, dto: LoginDto):
    print("Authenticating: ")
    print(dto)