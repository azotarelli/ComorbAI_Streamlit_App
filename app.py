import streamlit as st # Biblioteca Streamlit para criar a interface web
import pandas as pd
import unicodedata
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Funções de Pré-processamento (copiadas para consistência) ---
def limpar_texto_para_ia(texto_original):
    """
    Aplica as mesmas etapas de limpeza do ANTEC_PESSOAL_LIMPO do script de treinamento.
    """
    if pd.isna(texto_original):
        return ''
    
    temp_text = str(texto_original)
    
    # Substitui _x000D_ (case-insensitive) e qualquer outra quebra-liha por ponto (. )
    temp_text = re.sub(r'_x000d_|\n|\r|//', '. ', temp_text, flags=re.IGNORECASE)
    
    # Converte para letras minúsculas
    temp_text = temp_text.lower()
    
    # Remoção de acentuação
    temp_text = unicodedata.normalize('NFKD', temp_text).encode('ascii', 'ignore').decode('utf-8')
    
    # Substitui múltiplos espaços por um único espaço e remove espaços no início/fim.
    temp_text = re.sub(r'\s+', ' ', temp_text).strip()
    
    # Remove caracteres que não são letras, números, espaços ou pontuação chave.
    # Esta deve ser a ÚLTIMA etapa para remoção de caracteres.
    temp_text = re.sub(r'[^a-z0-9\s.,;:/]', '', temp_text)
    
    return temp_text

# --- Dicionários de Comorbidades Base (copiados para consistência) ---
# Necessário para saber quais comorbidades o modelo foi treinado para prever
COMORBIDADES_BASE_MAP = {
    'diabetes': ['diabetes', 'dm', 'dm1', 'dm2', 'diabetico', 'diabetica', 'glicemia alta', 'glicemia elevada', 'diabete', 'diabettes'],
    'hipertensao': ['hipertensao', 'has', 'hipertenso', 'hipertensa', 'pressao alta', 'aas'],
    'insuficiencia renal cronica': ['insuficiencia renal cronica', 'irc', 'doenca renal cronica', 'drc', 'dialise', 'creatinina alta', 'ureia alta'],
    'coronariopatia': ['coronariopatia', 'coronariopata', 'doenca coronariana', 'dac', 'iam', 'infarto', 'angina', 'cardiopatia', 'stent'],
    'transtornos mentais': ['transtornos mentais', 'depressao', 'ansiedade', 'esquizofrenia', 'esquizofenico', 'esquizofenica', 'bipolar', 'bipolaridade', 'psicose', 'saude mental'],
    'hipotireoidismo': ['hipotireoidismo', 'tireoide'],
    'insuficiencia cardiaca': ['insuficiencia cardiaca', 'icc', 'doenca cardiaca', 'insuficiencia cardiaca congestiva'],
    'demencia/alzheimer': ['demencia', 'alzheimer', 'demencia senil', 'parkinson'],
    'obesidade': ['obesidade', 'obeso', 'obesa', 'sobrepeso', 'imc alto'],
    'asma/dpoc': ['asma', 'asmatico', 'asmatica', 'dpoc', 'doenca pulmonar obstrutiva cronica', 'bronquite cronica', 'enfisema'],
    'acidente vascular encefalico/avc': ['acidente vascular encefalico', 'avc', 'isquemia cerebral', 'derrame'],
    'arritmia cardiaca': ['arritmia cardiaca', 'arritmia', 'fibrilacao atrial'],
    'osteoporose': ['osteoporose'],
    'marca passo': ['marca passo', 'marcapasso', 'marca-passo'],
    'glaucoma': ['glaucoma'],
    'dislipidemia': ['dislipidemia', 'colesterol alto', 'triglicerideos altos', 'dlp'],
    'endometriose': ['endometriose'],
    'nefrolitiase': ['nefrolitiase', 'pedra nos rins', 'calculo renal'],
    'transtornos globais do desenvolvimento/tea': ['transtornos globais do desenvolvimento', 'tea', 'autismo', 'autista', 'sindrome de asperger'],
    'oncologia': ['oncologia', 'cancer', 'neoplasia', 'tumor', 'metastase', 'quimio', 'quimioterapia', 'radioterapia'],
    'mastectomia': ['mastectomia'],
    'prostatectomia': ['prostatectomia'],
    'histerectomia': ['histerectomia'],
    'fumante': ['fumante', 'tabagista', 'tabagismo', 'cigarro', 'cigarros', 'fuma', 'fumo', 'nicotina', 'tabaco'],
    'etilismo': ['etilismo', 'alcoolismo', 'alcoolatra', 'dependencia de alcool', 'uso de alcool', 'beber', 'bebida alcoolica', 'alcool', 'etilista'],
    'sedentarismo': ['sedentarismo', 'sedentario', 'sedentaria', 'inatividade fisica', 'sem exercicio', 'pouca atividade']
}

# Lista de todas as comorbidades que o modelo foi treinado para prever
TODAS_COMORBIDADES = list(COMORBIDADES_BASE_MAP.keys())

# Mapeamento de índices para nomes de comorbidades (para exibir os resultados)
idx_to_comorb = {i: comorb for i, comorb in enumerate(TODAS_COMORBIDADES)}
NUM_LABELS = len(TODAS_COMORBIDADES)

# Configurações do modelo (devem ser as mesmas do treinamento)
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
MAX_LEN = 128
checkpoint_dir = 'comorb_ai_checkpoints'
checkpoint_filename = 'checkpoint_epoch_0.pth' # Use o checkpoint mais recente ou o final
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

# Detectar dispositivo (GPU ou CPU)
# Usamos st.cache_resource para carregar o modelo e o tokenizador apenas uma vez
# Isso é crucial para a performance da aplicação Streamlit
@st.cache_resource
def load_model_and_tokenizer():
    """
    Carrega o tokenizador e o modelo BERT do checkpoint.
    Esta função será executada apenas uma vez.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Carregando modelo no dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        st.write(f"Carregando estado do modelo do checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.write("Modelo carregado do checkpoint com sucesso!")
    else:
        st.error(f"Erro: Checkpoint '{checkpoint_path}' não encontrado.")
        st.stop() # Interrompe a execução se o checkpoint não for encontrado

    model.eval() # Coloca o modelo em modo de avaliação
    return tokenizer, model, device

# Carrega o modelo e o tokenizador
tokenizer, model, device = load_model_and_tokenizer()

# --- Interface do Streamlit ---
st.set_page_config(page_title="ComorbAI - Detector de Comorbidades", layout="centered")

st.title("🩺 ComorbAI - Detector de Comorbidades")
st.markdown("""
Esta ferramenta utiliza um modelo de Machine Learning (BERT) para analisar textos de anamnese
e identificar comorbidades relevantes.
""")

st.subheader("Digite o texto da anamnese abaixo:")
anamnese_input = st.text_area("Anamnese do Paciente", height=200, placeholder="Ex: Paciente com diabetes tipo 2, hipertenso, em uso de metformina e losartana. Relata histórico de infarto há 5 anos.")

if st.button("Analisar Anamnese"):
    if not anamnese_input:
        st.warning("Por favor, digite um texto para analisar.")
    else:
        with st.spinner("Analisando..."):
            # Limpar o texto de entrada
            texto_limpo_input = limpar_texto_para_ia(anamnese_input)

            if not texto_limpo_input:
                st.warning("Nenhum texto válido para analisar após a limpeza. Tente novamente.")
            else:
                # Tokenizar o texto de entrada
                encoding = tokenizer.encode_plus(
                    texto_limpo_input,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )

                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                # Fazer a previsão (inferência)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                st.subheader("Comorbidades Detectadas:")
                nenhuma_detectada = True
                for i, prob in enumerate(probs):
                    comorbidade = idx_to_comorb[i]
                    if prob > 0.5: # Limiar de 0.5 (50%)
                        st.success(f"- **{comorbidade.replace('_', ' ').title()}** (Probabilidade: {prob:.4f})")
                        nenhuma_detectada = False
                
                if nenhuma_detectada:
                    st.info("Nenhuma comorbidade confirmada detectada neste texto.")

st.markdown("---")
st.markdown("Desenvolvido para o projeto ComorbAI por Anderson Zotarelli.")

