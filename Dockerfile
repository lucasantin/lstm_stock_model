# Usar uma imagem base com Python 3.9
FROM python:3.9-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar os arquivos do projeto para dentro do contêiner
COPY . /app

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta que a API Flask vai rodar
EXPOSE 5000

# Definir o comando para iniciar a aplicação
CMD ["python", "app.py"]
