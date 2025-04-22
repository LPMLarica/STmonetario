import streamlit as st 
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objects as go 


DATA_INICIO ='2017-01-01'
DATA_FIM = date.today().strftime ('%Y-%m-%d')

st.title('Análise de açÕes')
st.sidebar.header('Escolha a ação')
n_dias = st.slider('Quantidade de dias de previsão', 30, 365)

def pegar_dados_acoes ( ):
    path = "C:\\Users\\joaop\\Downloads\\acoes.csv"
    return pd.read_csv(path, delimiter=';')

df = pegar_dados_acoes()
acao = df ['snome']

nome_acao_escolhido = st.sidebar.selectbox ('Escolha uma ação:', acao)

def_acao = df[ df['snome'] == nome_acao_escolhido]
acao_escolhida = def_acao.iloc[0] ['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'


@st.cache_data

def pegar_valores_online (sigla_acao):
    df = yf.download(sigla_acao, DATA_INICIO, DATA_FIM)
    df.reset_index(inplace=True)
    return df 

df_valores =pegar_valores_online(acao_escolhida)

st.subheader ('Tabela de Valores -' + nome_acao_escolhido)
st.write(df_valores.tail(10))

st.subheader ('Grafico de preços:')
fig= go.Figure ()
fig.add_trace( go.Scatter(x= df_valores['Date'],
                            y=df_valores[ 'Close'],
                            name='Preco Fechamento',
                            line_color= 'yellow'))
fig.add_trace( go.Scatter(x= df_valores['Date'],
                            y=df_valores[ 'Open'],
                            name='Abertura',
                            line_color= 'blue'))
st.plotly_chart(fig)

df_treino = df_valores[['Date', 'Close']]
df_treino = df_treino.rename(columns={"Date": 'ds', 'Close': 'y'})



modelo= Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe (periods=n_dias, freq='B')
previsao= modelo.predict(futuro)


st.subheader('Previsão')
st.write(previsao[['ds', 'yhat', 'yhat_low', 'yhat_upper']].tail(n_dias))

grafico1 = plot_plotly(modelo, previsao)
st.plotly_chart(grafico1)

grafico2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(grafico2)