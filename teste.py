import pandas as pd
import re
from unidecode import unidecode
from sqlalchemy import create_engine
import os

# Função para identificar intervalos no texto
def identificar_intervalo(texto):
    texto = unidecode(texto).lower()
    padroes = [
        r"entre\s*(\d+\.?\d*)\s*e\s*(\d+\.?\d*)",
        r"de\s*(\d+\.?\d*)\s*até\s*(\d+\.?\d*)",
        r"de\s*(\d+\.?\d*)\s*a\s*(\d+\.?\d*)",
        r"de\s*(\d+\.?\d*)\s*até\s*(\d+\.?\d*)",
        r"de\s*(\d+\.?\d*)\s*e\s*(\d+\.?\d*)",
        r"entre\s*(\d+\.?\d*)\s*e\s*(\d+\.?\d*)"
    ]
    
    for padrao in padroes:
        match = re.search(padrao, texto)
        if match:
            menor = float(match.group(1))
            maior = float(match.group(2))
            return menor, maior
    return None

# Função para conectar ao banco de dados
def conectar_banco_dados():
    DB_HOST = os.getenv('DB_HOST', 'db7.mepluga.com')
    DB_USER = os.getenv('DB_USER', 'usr_pilot')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'rCmLDygzzk7iBhOF')
    DB_DATABASE = os.getenv('DB_DATABASE', 'plug_copilot')
    
    database_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}'
    engine = create_engine(database_connection_str)
    df_produtos = pd.read_sql_table('produtos', con=engine)
    return df_produtos

# Função para filtrar produtos por intervalo de preço
def filtrar_produtos_por_intervalo(consulta):
    intervalo = identificar_intervalo(consulta)
    if not intervalo:
        print("Nenhum intervalo encontrado na consulta.")
        return None

    menor, maior = intervalo

    # Carregar dados dos produtos
    try:
        df_produtos = conectar_banco_dados()
    except Exception as e:
        print(f"Erro ao carregar dados do banco de dados: {e}")
        return None

    # Filtrar produtos pelo intervalo de preço
    df_filtrado = df_produtos[(df_produtos['preco'] >= menor) & (df_produtos['preco'] <= maior)]

    # Selecionar apenas as colunas desejadas
    df_filtrado = df_filtrado[['nome', 'descricao', 'preco']]

    return df_filtrado

# Exemplo de uso
consulta = "Me mostre produtos entre 80 e 120 reais."
df_resultado = filtrar_produtos_por_intervalo(consulta)

if df_resultado is not None:
    print(df_resultado)
else:
    print("Nenhum produto encontrado dentro do intervalo especificado.")
