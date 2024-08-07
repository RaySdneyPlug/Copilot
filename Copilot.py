import os
import pandas as pd
import json
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Variáveis de ambiente do banco de dados
DB_HOST = os.getenv('DB_HOST', 'db7.mepluga.com')
DB_USER = os.getenv('DB_USER', 'usr_pilot')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'rCmLDygzzk7iBhOF')
DB_DATABASE = os.getenv('DB_DATABASE', 'plug_copilot')

# Criação da string de conexão
database_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}'

def conectar_banco_dados():
    engine = create_engine(database_connection_str)
    df_produtos = pd.read_sql_table('produtos', con=engine)
    return df_produtos

def criar_pasta_train_data():
    pasta_train_data = os.path.join(os.path.dirname(__file__), 'train_data')
    
    if not os.path.exists(pasta_train_data):
        os.makedirs(pasta_train_data)
    
    return pasta_train_data

def normalizar_texto(texto):
    return texto.lower().strip()

def create_training_data(df_produtos):
    train_data = {"data": []}

    # Normalizar os nomes dos produtos no DataFrame
    df_produtos['nome_normalizado'] = df_produtos['nome'].apply(normalizar_texto)
    lista_produtos = df_produtos[['nome', 'preco', 'descricao', 'nome_normalizado']].to_dict(orient='records')

    # Encontrar o menor e o maior preço
    preco_mais_barato = df_produtos['preco'].min()
    produtos_mais_baratos = df_produtos[df_produtos['preco'] == preco_mais_barato]
    baratos = ', '.join(produtos_mais_baratos['nome'].tolist())

    preco_mais_caro = df_produtos['preco'].max()
    produtos_mais_caros = df_produtos[df_produtos['preco'] == preco_mais_caro]
    caros = ', '.join(produtos_mais_caros['nome'].tolist())

    # Dados de treino
    qa_pairs = [
        {
            "pergunta": [
                "Qual é o nome, o preço e a descrição do produto {nome}?",
                "Pode me dar informações sobre o produto {nome}?",
                "Me conte sobre o produto {nome}.",
                "Quero saber sobre o produto {nome}.",
                "Fale sobre o produto {nome}.",
                "O que é o produto {nome}?",
                "Detalhes sobre o produto {nome}.",
                "Informações sobre o produto {nome}, por favor.",
                "Me dê detalhes do produto {nome}.",
                "Conte-me sobre o produto {nome}."
            ],
            "resposta": "O produto {nome} custa {preco} reais e sua descrição é: {descricao}."
        },
        {
            "pergunta": [
                "Quais produtos custam até {valor} reais?",
                "Me mostre produtos com preço até {valor} reais.",
                "Quais são os produtos que custam no máximo {valor} reais?",
                "Produtos com preço até {valor} reais, por favor.",
                "Liste os produtos com preço até {valor} reais.",
                "Quais são os itens até {valor} reais?",
                "Produtos com preço menor ou igual a {valor} reais.",
                "Me diga quais produtos custam até {valor} reais.",
                "Quais são os itens abaixo de {valor} reais?",
                "Produtos que custam até {valor} reais."
            ],
            "resposta": "Os produtos com preço até {valor} reais são: {produtos_ate}."
        },
        {
            "pergunta": [
                "Qual é o produto mais barato?",
                "Me mostre produtos com preço mais baixo.",
                "Qual é o produto com o menor preço?",
                "Produto mais barato, por favor.",
                "Qual produto custa menos?",
                "Qual é o item mais barato?",
                "Produto com menor preço.",
                "Qual é o produto mais acessível?",
                "Qual produto tem o preço mais baixo?",
                "Me diga o produto mais barato."
            ],
            "resposta": "O(s) produto(s) mais barato(s) é(são): {baratos}."
        },
        {
            "pergunta": [
                "Qual é o produto mais caro?",
                "Me mostre produtos com preço mais alto.",
                "Qual é o produto com o maior preço?",
                "Produto mais caro, por favor.",
                "Qual produto custa mais?",
                "Qual é o item mais caro?",
                "Produto com maior preço.",
                "Qual é o produto mais caro disponível?",
                "Qual produto tem o preço mais alto?",
                "Me diga o produto mais caro."
            ],
            "resposta": "O(s) produto(s) mais caro(s) é(são): {caros}."
        },
        {
            "pergunta": [
                "Quais produtos você tem?",
                "Liste todos os produtos disponíveis.",
                "Quais são todos os produtos disponíveis?",
                "Me mostre todos os produtos.",
                "Quais itens estão disponíveis?",
                "Me dê a lista de produtos disponíveis.",
                "Liste todos os itens disponíveis.",
                "Quais são os produtos disponíveis para compra?",
                "Mostrar todos os produtos.",
                "Quais produtos estão no estoque?"
            ],
            "resposta": "Aqui está a lista de produtos: {tabela_produtos}."
        },
        {
            "pergunta": [
                "Quais produtos estão entre {min_valor} e {max_valor} reais?",
                "Me mostre produtos que custam entre {min_valor} e {max_valor} reais.",
                "Liste produtos com preço entre {min_valor} e {max_valor} reais.",
                "Quais itens estão entre {min_valor} e {max_valor} reais?",
                "Produtos entre {min_valor} e {max_valor} reais, por favor.",
                "Quais são os produtos com preço entre {min_valor} e {max_valor} reais?",
                "Mostre produtos entre {min_valor} e {max_valor} reais.",
                "Quais são os produtos com preço entre {min_valor} e {max_valor} reais?",
                "Produtos que custam entre {min_valor} e {max_valor} reais.",
                "Me diga quais produtos estão entre {min_valor} e {max_valor} reais."
            ],
            "resposta": "Os produtos com preço entre {min_valor} e {max_valor} reais são: {produtos_intervalo}."
        }
    ]

    # Adiciona perguntas e respostas para cada produto
    for product in lista_produtos:
        nome = product['nome']
        preco = product['preco']
        descricao = product['descricao']
        
        for pair in qa_pairs:
            for question in pair['pergunta']:
                valores = {
                    "nome": nome,
                    "preco": preco,
                    "descricao": descricao,
                    "valor": "",  # Adicionado para evitar erro
                    "produtos_ate": ', '.join([
                        f"{p['nome']} ({p['preco']} reais)"
                        for p in lista_produtos if p['preco'] <= 100
                    ]),
                    "tabela_produtos": '\n'.join([
                        f"{p['nome']}: {p['preco']} reais - {p['descricao']}"
                        for p in lista_produtos
                    ]),
                    "baratos": baratos,
                    "caros": caros,
                    "min_valor": 100,  # Definido como exemplo, deve ser ajustado conforme necessário
                    "max_valor": 200,  # Definido como exemplo, deve ser ajustado conforme necessário
                    "produtos_intervalo": ', '.join([
                        f"{p['nome']} ({p['preco']} reais)"
                        for p in lista_produtos if 100 <= p['preco'] <= 200
                    ])
                }

                resposta = pair['resposta'].format(**valores)
                train_data["data"].append({
                    "pergunta": question,
                    "resposta": resposta,
                    "paragraphs": [{"context": resposta}]
                })

    return train_data

def salvar_dados_treinamento(train_data, data_dir):
    train_data_path = os.path.join(data_dir, "train_data.json")
    if len(train_data["data"]) > 0:
        train_data_split, val_data_split = train_test_split(train_data["data"], test_size=0.2, random_state=42)

        with open(train_data_path, "w", encoding='utf-8') as f:
            json.dump({"data": train_data_split}, f, ensure_ascii=False, indent=4)
        print(f"Dados de treinamento salvos em {train_data_path}.")

        val_data_path = os.path.join(data_dir, "val_data.json")
        with open(val_data_path, "w", encoding='utf-8') as f:
            json.dump({"data": val_data_split}, f, ensure_ascii=False, indent=4)
        print(f"Dados de validação salvos em {val_data_path}.")
    else:
        print("Nenhum dado de treinamento foi gerado.")

def preprocessar_texto(texto):
    texto = texto.lower().strip()
    palavras = word_tokenize(texto)
    palavras = [p for p in palavras if p not in stopwords.words('portuguese')]
    return ' '.join(palavras)

def comparar_perguntas(pergunta_usuario, perguntas_treinamento):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(perguntas_treinamento + [pergunta_usuario])
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarity_matrix.flatten()

def lidar_com_saudacoes(pergunta):
    saudacoes = {
        "saudacoes": [
            "olá",
            "oi",
            "bom dia",
            "boa tarde",
            "boa noite",
            "e aí",
            "salve",
            "hey"
        ],
        "despedidas": [
            "tchau",
            "adeus",
            "até logo",
            "até mais",
            "até breve"
        ]
    }

    respostas = {
        "saudacoes": [
            "Olá! Como posso ajudar você hoje?",
            "Oi! Em que posso te ajudar?",
            "Bom dia! O que posso fazer por você?",
            "Boa tarde! O que você precisa?",
            "Boa noite! Posso te ajudar com algo?",
            "E aí! O que você está procurando?",
            "Salve! Estou à disposição para te ajudar.",
            "Hey! Como posso ser útil hoje?"
        ],
        "despedidas": [
            "Até logo! Se precisar, estarei por aqui.",
            "Adeus! Tenha um ótimo dia!",
            "Falou! Volte quando precisar.",
            "Até breve! Espero te ver novamente.",
            "Até já! Estarei por aqui se você precisar."
        ]
    }

    for padrao in saudacoes["saudacoes"]:
        if re.search(padrao, pergunta.lower()):
            return random.choice(respostas["saudacoes"])
    
    for padrao in saudacoes["despedidas"]:
        if re.search(padrao, pergunta.lower()):
            return random.choice(respostas["despedidas"])
    
    return None

def responder_com_base_no_produto(pergunta_usuario, df_produtos):
    pergunta_normalizada = preprocessar_texto(pergunta_usuario)
    for _, produto in df_produtos.iterrows():
        nome = produto['nome']
        nome_normalizado = produto['nome_normalizado']

        if nome_normalizado in pergunta_normalizada:
            return f"O produto {nome} custa {produto['preco']} reais e sua descrição é: {produto['descricao']}."
    
    return None

def responder_com_intervalo_de_preco(pergunta_usuario, df_produtos):
    match = re.search(r'produtos entre (\d+) e (\d+)', pergunta_usuario.lower())
    if match:
        min_valor, max_valor = map(int, match.groups())
        produtos_intervalo = [
            f"{p['nome']} ({p['preco']} reais)"
            for _, p in df_produtos.iterrows() if min_valor <= p['preco'] <= max_valor
        ]
        if produtos_intervalo:
            return f"Os produtos com preço entre {min_valor} e {max_valor} reais são: {', '.join(produtos_intervalo)}."
        else:
            return f"Não há produtos com preço entre {min_valor} e {max_valor} reais."
    return None

def fazer_pergunta(train_data, pergunta_usuario, df_produtos):
    try:
        pergunta_usuario = preprocessar_texto(pergunta_usuario)
        
        if 'data' not in train_data:
            return "Desculpe, dados de treinamento inválidos."
        
        perguntas_treinamento = [item['pergunta'] for item in train_data['data']]
        respostas_treinamento = [item['resposta'] for item in train_data['data']]
        
        similarity_scores = comparar_perguntas(pergunta_usuario, perguntas_treinamento)
        
        if len(similarity_scores) == 0:
            resposta_base_produto = responder_com_base_no_produto(pergunta_usuario, df_produtos)
            if resposta_base_produto:
                return resposta_base_produto
            resposta_intervalo_preco = responder_com_intervalo_de_preco(pergunta_usuario, df_produtos)
            if resposta_intervalo_preco:
                return resposta_intervalo_preco
            return "Desculpe, não entendi a sua pergunta. Pode tentar novamente?"

        max_similarity = np.max(similarity_scores)
        melhor_indice = np.argmax(similarity_scores)

        if max_similarity < 0.3:
            resposta_base_produto = responder_com_base_no_produto(pergunta_usuario, df_produtos)
            if resposta_base_produto:
                return resposta_base_produto
            resposta_intervalo_preco = responder_com_intervalo_de_preco(pergunta_usuario, df_produtos)
            if resposta_intervalo_preco:
                return resposta_intervalo_preco
            return "Desculpe, não entendi a sua pergunta. Pode tentar novamente?"

        resposta = respostas_treinamento[melhor_indice]
        return resposta
    except Exception as e:
        return f"Desculpe, não consegui entender a pergunta. Erro: {e}"

if __name__ == "__main__":
    df_produtos = conectar_banco_dados()

    if df_produtos.empty:
        print("O DataFrame de produtos está vazio. Verifique a conexão com o banco de dados e a tabela 'produtos'.")
    else:
        train_data = create_training_data(df_produtos)
        data_dir = criar_pasta_train_data()
        salvar_dados_treinamento(train_data, data_dir)

        while True:
            pergunta_usuario = input("Digite sua pergunta (ou 'sair' para encerrar): ")
            if pergunta_usuario.lower() == 'sair':
                break
            
            resposta_saudacao = lidar_com_saudacoes(pergunta_usuario)
            if resposta_saudacao:
                print(resposta_saudacao)
            else:
                resposta = fazer_pergunta(train_data, pergunta_usuario, df_produtos)
                print(resposta)
