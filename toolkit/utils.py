from datetime import datetime
from langchain_community.tools import Tool

def date_hour(query: str) -> str:
    return str(datetime.now())

date_hour_tool = Tool.from_function(
    func=date_hour,
    name="DateHourSearch",
    description="Para saber o dia e o horário atual"
)

def global_position(query: str) -> str:
    return "Teresina, Piauí, Brasil"

global_position_tool = Tool.from_function(
    func=global_position,
    name="GlobalPositionSearch",
    description="Para saber a cidade onde mora o usuário"
)
