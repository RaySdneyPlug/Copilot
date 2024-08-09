import pandas as pd
import re
from unidecode import unidecode
from sqlalchemy import create_engine
import os

class ProdutoFiltro:
    def __init__(self):
        self.engine = self.conectar_banco_dados()
    
    # Função para conectar ao banco de dados
    def conectar_banco_dados(self):
        DB_HOST = os.getenv('DB_HOST', 'db7.mepluga.com')
        DB_USER = os.getenv('DB_USER', 'usr_pilot')
        DB_PASSWORD = os.getenv('DB_PASSWORD', 'rCmLDygzzk7iBhOF')
        DB_DATABASE = os.getenv('DB_DATABASE', 'plug_copilot')
        
        database_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}'
        engine = create_engine(database_connection_str)
        return engine

    def carregar_produtos(self):
        try:
            df_produtos = pd.read_sql_table('produtos', con=self.engine)
            return df_produtos
        except Exception as e:
            print(f"Erro ao carregar dados do banco de dados: {e}")
            return None

    def identificar_intervalo(self, consulta):
        texto = unidecode(consulta).lower()
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

    def identificar_valor_maximo(self, texto):
        texto = unidecode(texto).lower()
        padroes = [
            r"ate\s*(\d+\.?\d*)",
            r"maximo\s*(\d+\.?\d*)",
            r"no maximo\s*(\d+\.?\d*)",
            r"ate\s*r?\$?\s*(\d+\.?\d*)",
            r"no maximo\s*r?\$?\s*(\d+\.?\d*)"
        ]
        
        for padrao in padroes:
            match = re.search(padrao, texto)
            if match:
                maximo = float(match.group(1))
                return maximo
        return None

    def identificar_valor_minimo(self, texto):
        texto = unidecode(texto).lower()
        padroes = [
            r"a partir de\s*(\d+\.?\d*)",
            r"minimo\s*(\d+\.?\d*)",
            r"no minimo\s*(\d+\.?\d*)",
            r"acima de\s*(\d+\.?\d*)",
            r"de\s*r?\$?\s*(\d+\.?\d*)\s*para cima"
        ]
        
        for padrao in padroes:
            match = re.search(padrao, texto)
            if match:
                minimo = float(match.group(1))
                return minimo
        return None

    def filtrar_produtos_por_intervalo(self, consulta):
        intervalo = self.identificar_intervalo(consulta)
        if not intervalo:
            return None

        menor, maior = intervalo
        df_produtos = self.carregar_produtos()

        if df_produtos is None:
            return None

        df_filtrado = df_produtos[(df_produtos['preco'] >= menor) & (df_produtos['preco'] <= maior)]
        df_filtrado = df_filtrado[['nome', 'descricao', 'preco']]

        return df_filtrado

    def filtrar_produtos_por_valor_maximo(self, consulta):
        maximo = self.identificar_valor_maximo(consulta)
        if maximo is None:
            return None

        df_produtos = self.carregar_produtos()
        if df_produtos is None:
            return None

        df_filtrado = df_produtos[df_produtos['preco'] <= maximo]
        df_filtrado = df_filtrado[['nome', 'descricao', 'preco']]

        return df_filtrado

    def filtrar_produtos_por_valor_minimo(self, consulta):
        minimo = self.identificar_valor_minimo(consulta)
        if minimo is None:
            return None

        df_produtos = self.carregar_produtos()
        if df_produtos is None:
            return None

        df_filtrado = df_produtos[df_produtos['preco'] >= minimo]
        df_filtrado = df_filtrado[['nome', 'descricao', 'preco']]

        return df_filtrado
