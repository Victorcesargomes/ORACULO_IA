import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from loaders import *
import tempfile
from langchain.prompts import ChatPromptTemplate

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'PDF', 'CSV', 'Txt'
]

CONFIG_MODELOS = {'Groq': {'modelos': ['llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
                           'chat': ChatGroq},
                  'OpenAI': {'modelos': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini'],
                             'chat': ChatOpenAI}}

MEMORIA = ConversationBufferWindowMemory(k=5)  # Mantém as últimas 5 mensagens



def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    
    if tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    
    if tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    
    if tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um Oráculo chamado Victor.
    Você detém conhecimento em diversas áreas do conhecimento.
    Você possui acesso às seguintes informações vindas de um documento {}:
    
    ###
    {}
    ###
    
    Utilize as informações fornecidas para baseas as suas respostas.
    
    Sempre que houver $ na sua saída, substitua por S.
    
    Se a informação  do documento  for algo como "Just a moment...Enable JavaScript and cookies to continue"
    Sugira carregar novamento o Oráculo!'''.format(tipo_arquivo, documento)
    tempate = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)

    chain = tempate | chat

    st.session_state['chain'] = chain
    

def pagina_chat():
    st.header('🤖 VR Engine IA', anchor=None)  # Corrigido

    chain = st.session_state.get('chain')

    if chain is None:
        st.error('Carregue o VR Engine IA')
        st.stop()
    memoria = st.session_state.get('memoria', MEMORIA)

    for mensagem in memoria.buffer_as_messages:
        chat_display = st.chat_message(mensagem.type)
        chat_display.markdown(mensagem.content)
    
    input_usuario = st.chat_input('Fale com o Oráculo')
    if input_usuario:
        
        chat_display = st.chat_message('human')
        chat_display.markdown(input_usuario)


        chat_display = st.chat_message('ai')
        resposta = chat_display.write_stream(chain.stream({'input': input_usuario, 'chat_history': memoria.buffer_as_messages}))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)  # Corrigido

        st.session_state['memoria'] = memoria
    

def sidebar():

    # Adicionar a logo no topo da barra lateral
    st.image("logo.png", use_container_width=True)  # Substitua pelo caminho da sua logo

    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo CSV', type=['.csv'])
        
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo Txt', type=['.txt'])
    
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        
        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Analista Contábil', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar Histórico de Conversas', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
        st.session_state['memoria'] = MEMORIA


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    


if __name__ == '__main__':
    main()
