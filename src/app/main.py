from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np


# Inicializar o aplicativo FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classe do modelo de entrada
class Dados(BaseModel):
    sem_fundamental: float
    sem_medio: float
    sem_superior: float
    analfabetismo: float
    ideb: float
    gini: float
    homicidios: float
    rendimentos_trabalho: float
    rendimento_domiciliar_per_capita: float
    bolsa_familia: float
    estado_Acre: float
    estado_Alagoas: float
    estado_Amapá: float
    estado_Amazonas: float
    estado_Bahia: float
    estado_Ceará: float
    estado_Distrito_Federal: float
    estado_Espírito_Santo: float
    estado_Goiás: float
    estado_Maranhão: float
    estado_Mato_Grosso: float
    estado_Mato_Grosso_do_Sul: float
    estado_Minas_Gerais: float
    estado_Paraná: float
    estado_Paraíba: float
    estado_Pará: float
    estado_Pernambuco: float
    estado_Piauí: float
    estado_Rio_Grande_do_Norte: float
    estado_Rio_Grande_do_Sul: float
    estado_Rio_de_Janeiro: float
    estado_Rondônia: float
    estado_Roraima: float
    estado_Santa_Catarina: float
    estado_Sergipe: float
    estado_São_Paulo: float
    estado_Tocantins: float
    ano_2016: float
    ano_2017: float
    ano_2018: float
    ano_2019: float
    ano_2020: float
    ano_2021: float
    ano_2022: float
    ano_2023: float

# Rota para previsão
@app.post("/previsao-total-vitimas/")
def previsao_total_vitimas(dados: Dados):
    # Criar DataFrame para o modelo
    input_data = pd.DataFrame({
        'sem_fundamental': [dados.sem_fundamental],
        'sem_medio': [dados.sem_medio],
        'sem_superior': [dados.sem_superior],
        'analfabetismo': [dados.analfabetismo],
        'ideb': [dados.ideb],
        'gini': [dados.gini],
        'homicidios': [dados.homicidios],
        'rendimentos_trabalho': [dados.rendimentos_trabalho],
        'rendimento_domiciliar_per_capita': [dados.rendimento_domiciliar_per_capita],
        'bolsa_familia': [dados.bolsa_familia],
        'estado_Acre': [dados.estado_Acre],
        'estado_Alagoas': [dados.estado_Alagoas],
        'estado_Amapá': [dados.estado_Amapá],
        'estado_Amazonas': [dados.estado_Amazonas],
        'estado_Bahia': [dados.estado_Bahia],
        'estado_Ceará': [dados.estado_Ceará],
        'estado_Distrito_Federal': [dados.estado_Distrito_Federal],
        'estado_Espírito_Santo': [dados.estado_Espírito_Santo],
        'estado_Goiás': [dados.estado_Goiás],
        'estado_Maranhão': [dados.estado_Maranhão],
        'estado_Mato_Grosso': [dados.estado_Mato_Grosso],
        'estado_Mato_Grosso_do_Sul': [dados.estado_Mato_Grosso_do_Sul],
        'estado_Minas_Gerais': [dados.estado_Minas_Gerais],
        'estado_Paraná': [dados.estado_Paraná],
        'estado_Paraíba': [dados.estado_Paraíba],
        'estado_Pará': [dados.estado_Pará],
        'estado_Pernambuco': [dados.estado_Pernambuco],
        'estado_Piauí': [dados.estado_Piauí],
        'estado_Rio_Grande_do_Norte': [dados.estado_Rio_Grande_do_Norte],
        'estado_Rio_Grande_do_Sul': [dados.estado_Rio_Grande_do_Sul],
        'estado_Rio_de_Janeiro': [dados.estado_Rio_de_Janeiro],
        'estado_Rondônia': [dados.estado_Rondônia],
        'estado_Roraima': [dados.estado_Roraima],
        'estado_Santa_Catarina': [dados.estado_Santa_Catarina],
        'estado_Sergipe': [dados.estado_Sergipe],
        'estado_São_Paulo': [dados.estado_São_Paulo],
        'estado_Tocantins': [dados.estado_Tocantins],
        'ano_2016': [dados.ano_2016],
        'ano_2017': [dados.ano_2017],
        'ano_2018': [dados.ano_2018],
        'ano_2019': [dados.ano_2019],
        'ano_2020': [dados.ano_2020],
        'ano_2021': [dados.ano_2021],
        'ano_2022': [dados.ano_2022],
        'ano_2023': [dados.ano_2023]
    })

    colunas_numericas = ['sem_fundamental', 'sem_medio', 'sem_superior', 'analfabetismo', 'ideb', 'gini',
                     'homicidios', 'rendimentos_trabalho', 'rendimento_domiciliar_per_capita',
                     'bolsa_familia', 'total_vitima']
    
    # Carregar o scaler salvo
    scaler = joblib.load('src/resources/scaler.pkl')

    # Carregar modelo e fazer previsão
    try:
        model = joblib.load('src/resources/model.pkl')
        previsao_normalizada = model.predict(input_data)

        # Seleciona o valor mínimo e máximo do total_vitima
        min_val = scaler.data_min_[colunas_numericas.index('total_vitima')]
        max_val = scaler.data_max_[colunas_numericas.index('total_vitima')]

        # Desnormalizando a previsão
        previsao_desnormalizada = previsao_normalizada * (max_val - min_val) + min_val

        print("teste2")

        # Encontrando o estado
        estado = None
        for col in input_data.columns:
            if "estado_" in col and input_data[col].iloc[0] == 1.0:  # Acessando a primeira linha de cada coluna
                estado = col.replace("estado_", "")  # Remove o prefixo "estado_"
                break

        # Encontrando o ano
        ano = None
        for col in input_data.columns:
            if "ano_" in col and input_data[col].iloc[0] == 1.0:  # Acessando a primeira linha de cada coluna
                ano = col.replace("ano_", "")  # Remove o prefixo "ano_"
                break

        # Exibindo a previsão desnormalizada
        return {f"A previsão do total aproximado de vítimas no estado {estado} no ano de {ano} é: {round(previsao_desnormalizada[0], 0)}"}
    except Exception as e:
        return {"error": str(e)}
