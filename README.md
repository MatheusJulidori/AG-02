# AG02

Este repositório contem o código do projeto realizado para matéria AG02 do Inatel no semestre 2023.2.
Desenvolvido por [Arthur Ferreira]((https://github.com/arthur-ngdi)) e [Matheus Julidori](https://github.com/MatheusJulidori).

## Preparando o ambiente

Para executar este projeto, é necessário um ambiente Python com as dependências instaladas. Para configurar o ambiente, siga as etapas abaixo:

1. Instale o Python em seu sistema, se ainda não estiver instalado. Você pode baixá-lo em [python.org](https://www.python.org/downloads/).

2. Crie um ambiente virtual (venv) para isolar as dependências do projeto. Você pode fazer isso com o seguinte comando:

    ```bash
    python -m venv venv
    ```
   
3. Ative o ambiente virtual. A maneira de ativar o ambiente varia dependendo do seu sistema operacional. Aqui estão alguns exemplos:
   
    No Windows:
     ```bash
     venv\Scripts\activate
      ```

    No macOS e Linux:
    
    ```bash
    source venv/bin/activate
    ```

4. Instale as dependências do projeto a partir do arquivo requirements.txt usando o comando pip:

    ```bash
    pip install -r requirements.txt
    ```

Agora seu ambiente virtual está configurado e as dependências do projeto estão instaladas. Você pode executar o projeto conforme necessário.

## Executando o projeto

Para executar o projeto, siga as etapas abaixo:

1. (Caso ja tenha ativado a venv, pule este passo) Ative o ambiente virtual. A maneira de ativar o ambiente varia dependendo do seu sistema operacional. Aqui estão alguns exemplos:
   
    No Windows:
     ```bash
     venv\Scripts\activate
      ```

    No macOS e Linux:
    
    ```bash
    source venv/bin/activate
    ```
   
2. Execute o arquivo main.py:

    ```bash
    python main.py
    ```
   
O programa irá exibir o resultado dos testes realizados e, depois, irá pedir a inserção de dados para serem testados.

**Legenda para inserção:**
- Quadrado marcado com X -> 1

- Quadrado marcado com O -> -1

- Quadrado vazio -> 0