import streamlit as st
import pandas as pd
import unicodedata
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from huggingface_hub import hf_hub_download # Importa a função para baixar do Hugging Face Hub

# --- Funções de Pré-processamento (mantidas) ---
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

# --- Dicionários de Comorbidades Base (mantidos) ---
COMORBIDADES_BASE_MAP = {
    'diabetes': ['diabetes', 'dm', 'dm1', 'dm2', 'diabetico', 'diabetica', 'glicemia alta', 'glicemia elevada', 'diabete', 'diabettes'],
    'hipertensao': ['hipertensao', 'has', 'hipertenso', 'hipertensa', 'pressao alta', 'aas'],
    'insuficiencia renal cronica': ['insuficiencia renal cronica', 'irc', 'doenca renal cronica', 'drc', 'dialise', 'creatinina alta', 'ureia alta'],
    'coronariopatia': ['coronariopatia', 'coronariopata', 'doenca coronariana', 'dac', 'iam', 'infarto', 'angina', 'cardiopatia', 'stent'],
    'transtornos mentais': ['transtornos mentais', 'depressao', 'ansiedade', 'esquizofrenia', 'esquizofenico', 'esquizofenica', 'bipolar', 'bipolaridade', 'psicose', 'saude mental'],
    'hipotireoidismo': ['hipotireoidismo', 'tireoide', 'Hipotireodismo'],
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
    'transtornos globais do desenvolvimento/tea': ['transtornos globais do desenvolvimento', 'tea', 'autismo', 'autista', 'sindrome de asperger', 'tdah'],
    'oncologia': ['oncologia', 'cancer', 'neoplasia', 'tumor', 'metastase', 'quimio', 'quimioterapia', 'radioterapia'],
    'mastectomia': ['mastectomia'],
    'prostatectomia': ['prostatectomia'],
    'histerectomia': ['histerectomia'],
    'fumante': ['fumante', 'tabagista', 'tabagismo', 'cigarro', 'cigarros', 'fuma', 'fumo', 'nicotina', 'tabaco'],
    'etilismo': ['etilismo', 'alcoolismo', 'alcoolatra', 'dependencia de alcool', 'uso de alcool', 'beber', 'bebida alcoolica', 'alcool', 'etilista'],
    'sedentarismo': ['sedentarismo', 'sedentario', 'sedentaria', 'inatividade fisica', 'sem exercicio', 'pouca atividade']
}

TODAS_COMORBIDADES = list(COMORBIDADES_BASE_MAP.keys())
idx_to_comorb = {i: comorb for i, comorb in enumerate(TODAS_COMORBIDADES)}
NUM_LABELS = len(TODAS_COMORBIDADES)

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
MAX_LEN = 128

# --- NOVO: Configuração para baixar do Hugging Face Hub ---
# Substitua 'seu_usuario' pelo seu nome de usuário no Hugging Face
# Substitua 'ComorbAI-Checkpoint' pelo nome do repositório que você criou para o modelo
HF_REPO_ID = "azotarelli/ComorbAI-Checkpoint" # Exemplo: "seu_usuario/ComorbAI-Checkpoint"
HF_FILENAME = "checkpoint_epoch_2.pth" # O nome do arquivo do checkpoint no Hugging Face Hub

@st.cache_resource
def load_model_and_tokenizer():
    """
    Carrega o tokenizador e o modelo BERT do checkpoint, baixando-o do Hugging Face Hub.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Carregando modelo no dispositivo: {device}")
    
    # Baixa o arquivo do Hugging Face Hub para o cache do Streamlit
    with st.spinner(f"Baixando modelo {HF_FILENAME} do Hugging Face Hub..."):
        try:
            checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
            st.write(f"Modelo baixado para: {checkpoint_path}")
        except Exception as e:
            st.error(f"Erro ao baixar o modelo do Hugging Face Hub: {e}")
            st.stop()

    st.write(f"Tentando carregar checkpoint de: {checkpoint_path}") # Linha de DEBUG

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        st.write(f"Carregando estado do modelo do checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.write("Modelo carregado do checkpoint com sucesso!")
    else:
        st.error(f"Erro: Checkpoint '{checkpoint_path}' não encontrado após download.")
        st.stop()

    model.eval()
    return tokenizer, model, device

# Carrega o modelo e o tokenizador
tokenizer, model, device = load_model_and_tokenizer()

# --- Interface do Streamlit (mantida) ---
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
            texto_limpo_input = limpar_texto_para_ia(anamnese_input)

            if not texto_limpo_input:
                st.warning("Nenhum texto válido para analisar após a limpeza. Tente novamente.")
            else:
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

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                st.subheader("Comorbidades Detectadas:")
                nenhuma_detectada = True
                for i, prob in enumerate(probs):
                    comorbidade = idx_to_comorb[i]
                    if prob > 0.5:
                        st.success(f"- **{comorbidade.replace('_', ' ').title()}** (Probabilidade: {prob:.4f})")
                        nenhuma_detectada = False
                
                if nenhuma_detectada:
                    st.info("Nenhuma comorbidade confirmada detectada neste texto.")

st.markdown("---")
st.markdown("Desenvolvido para o projeto ComorbAI. Por Anderson Zotarelli")