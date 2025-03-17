import requests
from bs4 import BeautifulSoup
from langchain_community.tools import Tool

CONTACTS_PAGE_URL = "https://www.trt22.jus.br/informes/agenda-de-contatos"  

def extract_from_contact_page(unit_name: str, column: int) -> str:
    try:
        # Faz a requisição HTTP para a página de contatos
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; LangchainBot/1.0)'}
        response = requests.get(CONTACTS_PAGE_URL, headers=headers, timeout=10)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Parseia o conteúdo HTML da página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Encontra todas as tabelas ou seções que contêm os contatos
        # (Ajuste o seletor conforme a estrutura da página)
        contact_tables = soup.find_all('table')  # Exemplo: busca todas as tabelas

        results = []
        for table in contact_tables:
            rows = table.find_all('tr')  # Encontra todas as linhas da tabela
            for row in rows:
                cells = row.find_all('td')  # Encontra todas as células da linha
                if len(cells) >= 1 + column:  # Assume que a primeira coluna é o nome e a segunda é o telefone
                    name = cells[0].get_text(strip=True).lower()
                    target = cells[column].get_text(strip=True)

                    # Verifica se o nome da unidade corresponde à consulta
                    if unit_name.lower() in name:
                        results.append(f"{name.capitalize()}: {target}")

        if results:
            return "\n".join(results)
        return f"Nenhum contato encontrado para '{unit_name}'."

    except requests.exceptions.RequestException as e:
        return f"Erro na requisição: {str(e)}"
    except Exception as e:
        return f"Erro: {str(e)}"

def extract_phone_number(unit_name: str) -> str:
    return extract_from_contact_page(unit_name, 1)

def extract_office_hours_number(unit_name: str) -> str:
    return extract_from_contact_page(unit_name, 2)

def get_content_page(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Erro ao acessar a página: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    texto = soup.get_text()
    
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    
    texto_limpo = '\n'.join(linhas)
    
    return texto_limpo

phone_extractor_tool = Tool.from_function(
    func=extract_phone_number,
    name="PhoneExtractor",
    description=(
        "Busca números de telefone de uma unidade administrativa "
        "na página de contatos do TRT22. Recebe o nome da unidade"
        " como entrada."
    )
)

office_hours_extractor_tool = Tool.from_function(
    func=extract_office_hours_number,
    name="OfficeHoursExtractor",
    description=(
        "Busca o horário de expediente de uma unidade administrativa "
        "na página de contatos do TRT22. Recebe o nome da unidade"
        " como entrada."
    )
)
