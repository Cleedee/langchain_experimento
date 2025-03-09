# class view-eventos-da-ejud
# class view-content
# class item-list
# ul li div.views-field-title
# span.field-content 
# a time 

from bs4 import BeautifulSoup
import requests

events_search_function_description = {
    'name': 'events_search',
    'description': '',
    'parameters': {},
}

def events_search() -> str:
    """
Retorna uma string cujo conteúdo é uma lista Python com eventos passados 
recentes e eventos futuros próximos na forma de dicionários Python. 
As chaves possuem os seguintes significados:

nome-evento: nome do evento
data-inicio: data do início do evento no formato dd/MM/YYYY
horário-inicio: hora que começa o evento
data-fim: data do fim do evento no formato dd/MM/YYYY
horario-fim: hora que encerra o evento 
pagina-evento: URL da página com informações sobre o evento
    """
    resultado = [
        {
            "nome-evento" : 'VERSÃO 4.8.1 GPREC E SUAS IMPLICAÇÕES NO AMBIENTE DE 2º GRAU - PJE - PRECATÓRIOS',
            "pagina-evento" : "https://www.trt22.jus.br/escola-judicial/evento/2025/versao-481-gprec-e-suas-implicacoes-no-ambiente-de-2o-grau-pje",
            "data-inicio": "20/02/2025",
            "horario-inicio": "09:00",
            "data-fim": "27/02/2025",
            "horario-fim": "17:00"
        }
    ]
    return str(resultado)

toolkit = [
    {
        'type': 'function',
        'function': events_search_function_description
    }
]

if __name__ == '__main__':
    print(events_search())
