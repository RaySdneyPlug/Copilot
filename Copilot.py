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
import unicodedata
from datetime import datetime
nltk.download('punkt')
nltk.download('stopwords')
from Perg_precos import ProdutoFiltro


Filtropreco = ProdutoFiltro()
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

    # Função para formatar a lista de produtos
    def formatar_lista_produtos(lista_produtos):
        linhas = []
        for p in lista_produtos:
            nome = p['nome']
            preco = p['preco']
            descricao = p['descricao']
            linhas.append(f"{nome:<20} | {preco:>8.2f} reais | {descricao}")

        tabela = "\n".join(linhas)
        return tabela

    # Criar perguntas e respostas para cada produto
    for produto in lista_produtos:
        nome = produto['nome']
        preco = produto['preco']
        descricao = produto['descricao']

        # Perguntas e respostas relacionadas a cada produto
        qa_pairs = [
            {
                "pergunta": [
                    f"Qual é o nome, o preço e a descrição do produto {nome}?",
                    f"Pode me dar informações sobre o produto {nome}?",
                    f"Me conte sobre o produto {nome}.",
                    f"Quero saber sobre o produto {nome}.",
                    f"Fale sobre o produto {nome}.",
                    f"O que é o produto {nome}?",
                    f"Detalhes sobre o produto {nome}.",
                    f"Informações sobre o produto {nome}, por favor.",
                    f"Me dê detalhes do produto {nome}.",
                    f"Conte-me sobre o produto {nome}."
                ],
                "resposta": f"O produto {nome} custa {preco:.2f} reais e sua descrição é: {descricao}.",
                "paragraphs": [{"context": f"O produto {nome} custa {preco:.2f} reais e sua descrição é: {descricao}."}]
            },
            {
                "pergunta": [
                    f"Quais produtos custam até {preco:.2f} reais?",
                    f"Me mostre produtos com preço até {preco:.2f} reais.",
                    f"Quais são os produtos que custam no máximo {preco:.2f} reais?",
                    f"Produtos com preço até {preco:.2f} reais, por favor.",
                    f"Liste os produtos com preço até {preco:.2f} reais.",
                    f"Quais são os itens até {preco:.2f} reais?",
                    f"Produtos com preço menor ou igual a {preco:.2f} reais.",
                    f"Me diga quais produtos custam até {preco:.2f} reais.",
                    f"Quais são os itens abaixo de {preco:.2f} reais?",
                    f"Produtos que custam até {preco:.2f} reais."
                ],
                "resposta": f"Os produtos com preço até {preco:.2f} reais são:\n{formatar_lista_produtos([p for p in lista_produtos if p['preco'] <= preco])}.",
                "paragraphs": [{"context": f"Os produtos com preço até {preco:.2f} reais são:\n{formatar_lista_produtos([p for p in lista_produtos if p['preco'] <= preco])}."}]
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
                "resposta": f"O(s) produto(s) mais barato(s) é(são): {baratos}.",
                "paragraphs": [{"context": f"O(s) produto(s) mais barato(s) é(são): {baratos}."}]
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
                "resposta": f"O(s) produto(s) mais caro(s) é(são): {caros}.",
                "paragraphs": [{"context": f"O(s) produto(s) mais caro(s) é(são): {caros}."}]
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
                "resposta": f"Aqui está a lista de produtos:\n\n{formatar_lista_produtos(lista_produtos)}.",
                "paragraphs": [{"context": f"Aqui está a lista de produtos:\n\n{formatar_lista_produtos(lista_produtos)}."}]
            },
        ]

        # Adiciona cada par de pergunta e resposta ao treinamento
        for pair in qa_pairs:
            for question in pair['pergunta']:
                train_data["data"].append({
                    "pergunta": question,
                    "resposta": pair['resposta'],
                    "paragraphs": pair['paragraphs']
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
    saudacoes = [
        "olá", "oi", "bom dia", "boa tarde", "boa noite", "e aí", "salve", "hey", "chat", "alô", "saudações", "oi oi",
        "ola", "ola!", "olá!", "olá, como vai?", "bom dia!", "boa tarde!", "boa noite!", "oi, tudo bem?", "saudações!", "hey, tudo bem?",
        "oi, como está?", "olá, tudo bem?", "oi, olá", "como vai?", "como você está?", "e aí, tudo certo?", "e aí, chat", "e aí, robô",
        "salve, chat", "salve, robô", "olá, chat", "oi, robô", "e aí, pessoal","eai"]
    
    despedidas = [
        "tchau", "adeus", "até logo", "até mais", "até breve", "falou", "nos vemos", "até a próxima", "até mais ver", "adeus, até mais",
        "até logo mais", "um abraço", "um beijo", "até logo!", "até breve!", "nos vemos em breve", "fique bem", "boa sorte", "cuide-se",
        "até mais!", "até logo, chat", "até breve, robô", "até mais, pessoal", "tchau, chat", "tchau, robô", "um abraço, chat", "um beijo, robô", "valeu"
    ]

    respostas_saudacoes = [
        "Olá! Como posso ajudar você hoje?",
        "Oi! Em que posso te ajudar?",
        "Bom dia! O que posso fazer por você?",
        "Boa tarde! O que você precisa?",
        "Boa noite! Posso te ajudar com algo?",
        "E aí! O que você está procurando?",
        "Estou à disposição para te ajudar.",
        "Hey! Como posso ser útil hoje?",
        "Saudações! Como posso ajudar?"
    ]
    
    respostas_despedidas = [
        "Até logo! Se precisar, estarei por aqui.",
        "Adeus! Tenha um ótimo dia!",
        "Falou! Volte quando precisar.",
        "Até breve! Espero te ver novamente.",
        "Até já! Estarei por aqui se você precisar.",
        "Um abraço! Volte sempre.",
        "Fique bem! Até a próxima.",
        "Cuide-se! Até logo."
    ]

    hora_atual = datetime.now().hour
    if 5 <= hora_atual < 12:
        saudacao_periodo = "Bom dia!"
    elif 12 <= hora_atual < 18:
        saudacao_periodo = "Boa tarde!"
    else:
        saudacao_periodo = "Boa noite!"

    # Normalizar a pergunta
    pergunta_normalizada = pergunta.lower().strip()

    # Verificar saudações
    for saudacao in saudacoes:
        if pergunta_normalizada.startswith(saudacao):
            return random.choice([resposta for resposta in respostas_saudacoes if saudacao_periodo in resposta])


    # Verificar despedidas
    for despedida in despedidas:
        if despedida in pergunta_normalizada:
            return random.choice(respostas_despedidas)
    
    return None

def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def responder_com_base_no_produto(pergunta_usuario, df_produtos):
    pergunta_normalizada = remover_acentos(preprocessar_texto(pergunta_usuario))
    
    for _, produto in df_produtos.iterrows():
        nome = remover_acentos(produto['nome']).lower()
        nome_normalizado = produto['nome_normalizado']

        if nome_normalizado in pergunta_normalizada:
            return f"O produto {produto['nome']} custa {produto['preco']} reais e sua descrição é: {produto['descricao']}."
    
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
            return "Desculpe, não entendi a sua pergunta. Pode tentar novamente?"

        max_similarity = np.max(similarity_scores)
        melhor_indice = np.argmax(similarity_scores)

        if max_similarity < 0.3:
            resposta_base_produto = responder_com_base_no_produto(pergunta_usuario, df_produtos)
            if resposta_base_produto:
                return resposta_base_produto
            return "Desculpe, não entendi a sua pergunta. Pode tentar novamente?"

        resposta = respostas_treinamento[melhor_indice]
        return resposta
    except Exception as e:
        return f"Desculpe, não consegui entender a pergunta. Erro: {e}"
    
def main():
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
                continue
            
            # Primeiro, verifique a filtragem de preços
            resposta_maximo = Filtropreco.filtrar_produtos_por_valor_maximo(pergunta_usuario)
            if resposta_maximo is not None and not resposta_maximo.empty:
                print(resposta_maximo)
                continue

            resposta_minimo = Filtropreco.filtrar_produtos_por_valor_minimo(pergunta_usuario)
            if resposta_minimo is not None and not resposta_minimo.empty:
                print(resposta_minimo)
                continue

            resposta_intervalo = Filtropreco.filtrar_produtos_por_intervalo(pergunta_usuario)
            if resposta_intervalo is not None and not resposta_intervalo.empty:
                print(resposta_intervalo)
                continue

            # Se nenhuma das condições anteriores for atendida, use o modelo treinado
            resposta = fazer_pergunta(train_data, pergunta_usuario, df_produtos)
            print(resposta)

if __name__ == "__main__":
    main()
